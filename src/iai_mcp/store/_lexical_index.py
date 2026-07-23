"""Persistent plaintext BM25 index over decrypted record surfaces."""
from __future__ import annotations

import functools
import json
import math
import os
import re
import tempfile
import threading
import unicodedata
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

_RAW_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_STOP_FR = frozenset(
    "le la les un une des de du d l et ou a au aux en dans sur pour par que qui "
    "quoi quel quelle quels quelles quand comment pourquoi ce cet cette ces on se "
    "je tu il elle nous vous ils elles est sont ete etre y c s n ne pas plus si "
    "avec sans mon ma mes ton ta tes son sa tres bien fait faire va vais dois "
    "peux peut ca cela meme entre vers depuis apres avant encore deja".split()
)

MAX_QUERY_TOKENS = 64
BM25_K1 = 1.5
BM25_B = 0.75
HYBRID_IDF_GATE = 4.0
RRF_K = 240
MIN_HYBRID_IDF_GATE = 1.0
MAX_HYBRID_IDF_GATE = 10.0
MIN_RRF_K = 10.0
MAX_RRF_K = 1000.0
INDEX_VERSION = 1
INDEX_FILENAME = "lexical-index.jsonl"


def _bounded_env_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return min(maximum, max(minimum, value))


def hybrid_rrf_k() -> float:
    """Return the bounded RRF constant configured for this recall."""
    return _bounded_env_float(
        "IAI_MCP_HYBRID_RRF_K",
        float(RRF_K),
        MIN_RRF_K,
        MAX_RRF_K,
    )


def hybrid_idf_gate() -> float:
    """Return the bounded rare-token IDF threshold."""
    return _bounded_env_float(
        "IAI_MCP_HYBRID_IDF_GATE",
        HYBRID_IDF_GATE,
        MIN_HYBRID_IDF_GATE,
        MAX_HYBRID_IDF_GATE,
    )


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """French-normalized words plus whole and split snake/camel identifiers."""
    out: list[str] = []
    for raw in _RAW_TOKEN_RE.findall(text or ""):
        whole = _normalize(raw).strip("_")
        if len(whole) > 1 and whole not in _STOP_FR:
            out.append(whole)
        parts = re.split(r"_+", raw)
        parts = [
            piece
            for part in parts
            for piece in _CAMEL_RE.split(part)
        ]
        if len(parts) > 1:
            for part in parts:
                token = _normalize(part)
                if len(token) > 1 and token not in _STOP_FR:
                    out.append(token)
    return out


@functools.lru_cache(maxsize=1)
def synonym_map() -> dict[str, tuple[str, ...]]:
    """Load the deterministic French query-expansion map from package data."""
    raw = json.loads(
        resources.files("iai_mcp")
        .joinpath("data/hybrid_synonyms_fr.json")
        .read_text(encoding="utf-8")
    )
    return {
        _normalize(key): tuple(
            token
            for value in values
            for token in tokenize(str(value))
        )
        for key, values in raw.items()
    }


def expand_query_tokens(text: str) -> list[str]:
    base = tokenize(text)
    synonyms = synonym_map()
    expanded = list(base)
    for token in base:
        expanded.extend(synonyms.get(token, ()))
    return list(dict.fromkeys(expanded))[:MAX_QUERY_TOKENS]


def reciprocal_rank_fusion(
    semantic_ids: Iterable[Any],
    lexical_ids: Iterable[Any],
    *,
    k: float = RRF_K,
) -> dict[Any, float]:
    """Fuse two ranked lanes without comparing their incompatible scores."""
    fused: dict[Any, float] = {}
    for lane in (semantic_ids, lexical_ids):
        for rank, record_id in enumerate(lane, start=1):
            fused[record_id] = fused.get(record_id, 0.0) + 1.0 / (k + rank)
    return fused


class LexicalIndex:
    """Thread-safe resident BM25 index with a plaintext JSONL sidecar."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_len: dict[str, int] = {}
        self._avg_len = 1.0
        self._n_docs = 0
        self._generation: Any = None
        self._ready = False
        self._lock = threading.RLock()
        self.build_lock = threading.Lock()

    @property
    def generation(self) -> Any:
        return self._generation

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def document_count(self) -> int:
        return self._n_docs

    def invalidate(self) -> None:
        """Make recall degrade to semantic-only until the next warm rebuild."""
        with self._lock:
            self._ready = False

    def _install(self, docs: dict[str, list[str]], generation: Any) -> None:
        postings: dict[str, dict[str, int]] = {}
        for rid, tokens in docs.items():
            for token in tokens:
                bucket = postings.setdefault(token, {})
                bucket[rid] = bucket.get(rid, 0) + 1
        with self._lock:
            self._postings = postings
            self._doc_tokens = docs
            self._doc_len = {rid: len(tokens) for rid, tokens in docs.items()}
            self._n_docs = len(docs)
            self._avg_len = max(
                sum(self._doc_len.values()) / self._n_docs if self._n_docs else 1.0,
                1.0,
            )
            self._generation = generation
            self._ready = True

    def load(self, generation: Any = None) -> bool:
        if self.path is None or not self.path.exists():
            return False
        docs: dict[str, list[str]] = {}
        try:
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    if "version" in item:
                        if item["version"] != INDEX_VERSION:
                            return False
                        continue
                    rid = str(item["id"])
                    tokens = item.get("tokens")
                    if tokens is None:
                        docs.pop(rid, None)
                    else:
                        docs[rid] = [str(token) for token in tokens]
        except (OSError, ValueError, KeyError, TypeError):
            return False
        self._install(docs, generation)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return True

    def build(self, rows: list[tuple[str, str]], generation: Any = None) -> None:
        docs = {str(rid): tokenize(surface) for rid, surface in rows}
        self._install(docs, generation)
        if self.path is not None:
            self._write_snapshot()

    def _write_snapshot(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            docs = dict(self._doc_tokens)
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                tmp_name = handle.name
                handle.write(json.dumps({"version": INDEX_VERSION}) + "\n")
                for rid, tokens in sorted(docs.items()):
                    handle.write(
                        json.dumps(
                            {"id": rid, "tokens": tokens},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def _append(self, rid: str, tokens: list[str] | None) -> None:
        if self.path is None:
            return
        if not self.path.exists():
            self._write_snapshot()
            return
        line = json.dumps(
            {"id": rid, "tokens": tokens},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        fd = os.open(self.path, os.O_APPEND | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        os.chmod(self.path, 0o600)

    def upsert(self, record_id: str, surface: str) -> bool:
        tokens = tokenize(surface)
        rid = str(record_id)
        with self._lock:
            if not self._ready:
                return False
            old = self._doc_tokens.get(rid, [])
            for token in old:
                bucket = self._postings.get(token)
                if bucket is None:
                    continue
                bucket.pop(rid, None)
                if not bucket:
                    self._postings.pop(token, None)
            self._doc_tokens[rid] = tokens
            self._doc_len[rid] = len(tokens)
            for token in tokens:
                bucket = self._postings.setdefault(token, {})
                bucket[rid] = bucket.get(rid, 0) + 1
            self._n_docs = len(self._doc_tokens)
            self._avg_len = max(
                sum(self._doc_len.values()) / self._n_docs if self._n_docs else 1.0,
                1.0,
            )
            self._append(rid, tokens)
        return True

    def remove(self, record_id: str) -> bool:
        rid = str(record_id)
        with self._lock:
            if not self._ready or rid not in self._doc_tokens:
                return False
            for token in self._doc_tokens.pop(rid):
                bucket = self._postings.get(token)
                if bucket is None:
                    continue
                bucket.pop(rid, None)
                if not bucket:
                    self._postings.pop(token, None)
            self._doc_len.pop(rid, None)
            self._n_docs = len(self._doc_tokens)
            self._avg_len = max(
                sum(self._doc_len.values()) / self._n_docs if self._n_docs else 1.0,
                1.0,
            )
            self._append(rid, None)
        return True

    @staticmethod
    def _bm25(tf: int, df: int, length: int, n_docs: int, avg_len: float) -> float:
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length / avg_len)
        return idf * tf * (BM25_K1 + 1.0) / denom

    def _idf(self, token: str) -> float:
        posting = self._postings.get(token)
        if not posting or not self._n_docs:
            return 0.0
        return math.log(
            1.0 + (self._n_docs - len(posting) + 0.5) / (len(posting) + 0.5)
        )

    def _query_tokens(self, tokens: list[str], k: int) -> list[tuple[str, float]]:
        tokens = list(dict.fromkeys(tokens))[:MAX_QUERY_TOKENS]
        if not tokens:
            return []
        postings = [self._postings[token] for token in tokens if token in self._postings]
        if not postings:
            return []
        ids = set().union(*postings)
        scored = [
            (
                rid,
                sum(
                    self._bm25(
                        posting[rid],
                        len(posting),
                        self._doc_len.get(rid, 1),
                        self._n_docs,
                        self._avg_len,
                    )
                    for posting in postings
                    if rid in posting
                ),
            )
            for rid in ids
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:k]

    def query(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        with self._lock:
            return self._query_tokens(tokenize(text), k)

    def gated_query(
        self,
        text: str,
        k: int = 10,
        *,
        threshold: float = HYBRID_IDF_GATE,
    ) -> tuple[bool, list[tuple[str, float]], float]:
        tokens = expand_query_tokens(text)
        with self._lock:
            max_idf = max((self._idf(token) for token in tokens), default=0.0)
            if max_idf < threshold:
                return False, [], max_idf
            return True, self._query_tokens(tokens, k), max_idf
