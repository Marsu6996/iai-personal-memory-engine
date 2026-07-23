from __future__ import annotations

import os
import stat
from uuid import uuid4

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.store._lexical_index import (
    HYBRID_IDF_GATE,
    LexicalIndex,
    hybrid_idf_gate,
    hybrid_rrf_k,
    reciprocal_rank_fusion,
)


def _idf_corpus(target_text: str) -> list[tuple[str, str]]:
    return [("target", target_text)] + [
        (f"noise-{i}", f"commun remplissage document {i}")
        for i in range(99)
    ]


def test_idf_gate_skips_banal_query_without_bm25(monkeypatch):
    idx = LexicalIndex()
    idx.build(_idf_corpus("commun information ordinaire"))
    monkeypatch.setattr(
        idx,
        "_query_tokens",
        lambda *_args, **_kwargs: pytest.fail("BM25 must not run below the IDF gate"),
    )

    gated, hits, max_idf = idx.gated_query("commun")

    assert gated is False
    assert hits == []
    assert max_idf < HYBRID_IDF_GATE


def test_synonym_expansion_versionnage_finds_semver():
    idx = LexicalIndex()
    idx.build(_idf_corpus("Le projet adopte semver pour chaque release"))

    gated, hits, max_idf = idx.gated_query("Quelle règle de versionnage ?")

    assert gated is True
    assert max_idf >= HYBRID_IDF_GATE
    assert hits[0][0] == "target"


def test_rrf_fuses_semantic_and_lexical_lanes_without_filtering():
    fused = reciprocal_rank_fusion(
        ["semantic-top", "agreed", "lexical-top"],
        ["lexical-top", "agreed", "lexical-only"],
    )

    assert set(fused) == {
        "semantic-top",
        "agreed",
        "lexical-top",
        "lexical-only",
    }
    assert fused["lexical-top"] > fused["semantic-top"]
    assert fused["agreed"] > fused["lexical-only"]


def test_hybrid_env_overrides_are_bounded_and_robust(monkeypatch):
    monkeypatch.setenv("IAI_MCP_HYBRID_RRF_K", "120")
    monkeypatch.setenv("IAI_MCP_HYBRID_IDF_GATE", "5.5")
    assert hybrid_rrf_k() == 120.0
    assert hybrid_idf_gate() == 5.5

    monkeypatch.setenv("IAI_MCP_HYBRID_RRF_K", "9999")
    monkeypatch.setenv("IAI_MCP_HYBRID_IDF_GATE", "0.1")
    assert hybrid_rrf_k() == 1000.0
    assert hybrid_idf_gate() == 1.0

    monkeypatch.setenv("IAI_MCP_HYBRID_RRF_K", "invalid")
    monkeypatch.setenv("IAI_MCP_HYBRID_IDF_GATE", "nan")
    assert hybrid_rrf_k() == 240.0
    assert hybrid_idf_gate() == HYBRID_IDF_GATE


def test_plaintext_index_is_incremental_persistent_and_mode_600(tmp_path):
    path = tmp_path / "lexical-index.jsonl"
    idx = LexicalIndex(path)
    idx.build([("first", "alpha")])
    idx.upsert("second", "marqueur incremental rare")

    loaded = LexicalIndex(path)
    assert loaded.load() is True
    assert loaded.query("incremental")[0][0] == "second"
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_capture_updates_warm_index_without_rebuild(tmp_path):
    store = MemoryStore(path=tmp_path)
    store.warm_lexical_index()
    result = capture_turn(
        store,
        cue="",
        text="Le marqueur IncrementaliteLexicale confirme la capture.",
        tier="episodic",
        session_id="m2-test",
        role="user",
    )
    flush_record_buffer(store)

    hits = store.lexical_search("IncrementaliteLexicale")

    assert str(hits[0][0].id) == result["record_id"]
    assert store._lexical_idx.document_count == 1


def test_kill_switch_disables_hybrid_lane(tmp_path, monkeypatch):
    store = MemoryStore(path=tmp_path)
    rare_id = uuid4()
    store._lexical_idx.build(
        [(str(rare_id), "semver")] + [
            (str(uuid4()), f"commun document {i}") for i in range(99)
        ]
    )
    monkeypatch.setenv("IAI_MCP_HYBRID_LEXICAL", "0")

    gated, hits, max_idf = store.hybrid_lexical_search_ids("versionnage")

    assert gated is False
    assert hits == []
    assert max_idf == 0.0
