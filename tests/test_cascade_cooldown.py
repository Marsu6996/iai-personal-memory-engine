from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


def test_at_most_six_cascades_over_five_minute_window_with_continuous_pending(monkeypatch):
    asyncio.run(_at_most_six_cascades_body(monkeypatch))


async def _at_most_six_cascades_body(monkeypatch):
    import iai_mcp.daemon as daemon_mod

    cascade_invocations: list[float] = []
    sentinel_assignment = type("Asgmt", (), {"top_communities": [], "mid_regions": {}})()

    clock = [1000.0]

    def fake_monotonic():
        return clock[0]

    def counting_stub(store):
        cascade_invocations.append(fake_monotonic())
        return (sentinel_assignment, [], 0, "normal")

    async def fast_cascade_stub(store, assignment, **kwargs):
        return {"communities_selected": 0, "records_warmed": 0}

    state_holder = {
        "fsm_state": "WAKE",
        "hippea_cascade_request": {"pending": True, "session_id": "test"},
    }

    def load_state_stub():
        return dict(state_holder)

    def save_state_stub(state):
        state_holder.update(state)
        state_holder["hippea_cascade_request"] = {
            "pending": True, "session_id": "test",
        }

    def write_event_stub(*args, **kwargs):
        return None

    monkeypatch.setattr(daemon_mod, "_last_cascade_completed_at", 0.0)
    monkeypatch.setattr(daemon_mod, "HIPPEA_CASCADE_POLL_SEC", 0.05)

    shutdown = asyncio.Event()

    with patch("iai_mcp.runtime_graph_cache.load_recall_structural", counting_stub), \
         patch("iai_mcp.hippea_cascade.run_cascade", fast_cascade_stub), \
         patch("iai_mcp.daemon_state.load_state", load_state_stub), \
         patch("iai_mcp.daemon_state.save_state", save_state_stub), \
         patch("iai_mcp.daemon.write_event", write_event_stub):

        cascade_task = asyncio.create_task(
            daemon_mod._hippea_cascade_loop(
                store=None, shutdown=shutdown, _clock=fake_monotonic,
            ),
        )

        POLL_STEP = 5.0
        WINDOW = 300.0
        steps = int(WINDOW / POLL_STEP)
        for _ in range(steps):
            clock[0] += POLL_STEP
            await asyncio.sleep(0.02)

        shutdown.set()
        try:
            await asyncio.wait_for(cascade_task, timeout=2.0)
        except asyncio.TimeoutError:
            cascade_task.cancel()
            try:
                await cascade_task
            except (asyncio.CancelledError, Exception):
                pass

        n = len(cascade_invocations)
        assert n <= 6, (
            f"R2 FAIL: {n} cascade invocations in 5-min window with "
            f"continuous pending=true. Expected ≤ 6 with 60s cooldown."
        )
        assert n >= 2, (
            f"R2 FAIL: only {n} cascade invocations across simulated "
            f"5-min window. Expected ≥ 2 (cooldown should release after "
            f"60 simulated seconds). Test fixture / mocks broken."
        )


def test_cooldown_clears_after_min_interval_elapsed():
    asyncio.run(_cooldown_clears_after_min_interval_body())


async def _cooldown_clears_after_min_interval_body():
    import iai_mcp.daemon as daemon_mod

    clock = [1000.0]

    def fake_monotonic():
        return clock[0]

    with patch("iai_mcp.daemon.time.monotonic", fake_monotonic):
        daemon_mod._last_cascade_completed_at = 1000.0
        elapsed = fake_monotonic() - daemon_mod._last_cascade_completed_at
        assert elapsed < daemon_mod.HIPPEA_CASCADE_MIN_INTERVAL_SEC

        clock[0] = 1000.0 + daemon_mod.HIPPEA_CASCADE_MIN_INTERVAL_SEC + 0.1
        elapsed = fake_monotonic() - daemon_mod._last_cascade_completed_at
        assert elapsed >= daemon_mod.HIPPEA_CASCADE_MIN_INTERVAL_SEC


def test_cascade_clears_canonical_state_without_disk_resurrection(monkeypatch):
    asyncio.run(_cascade_clears_canonical_state_body(monkeypatch))


async def _cascade_clears_canonical_state_body(monkeypatch):
    import iai_mcp.daemon as daemon_mod

    state = {
        "fsm_state": "WAKE",
        "hippea_cascade_request": {"pending": True, "session_id": "canonical"},
    }
    calls: list[int] = []
    shutdown = asyncio.Event()

    def load_structural_stub(store):
        calls.append(1)
        return None, [], 0, "cold_degrade"

    def save_state_stub(saved):
        assert saved["hippea_cascade_request"] == {"pending": False}

    monkeypatch.setattr(daemon_mod, "_last_cascade_completed_at", 0.0)
    monkeypatch.setattr(daemon_mod, "HIPPEA_CASCADE_POLL_SEC", 0.01)

    with patch("iai_mcp.runtime_graph_cache.load_recall_structural", load_structural_stub), \
         patch("iai_mcp.daemon_state.load_state", side_effect=AssertionError("disk state must not be reloaded")), \
         patch("iai_mcp.daemon_state.save_state", save_state_stub), \
         patch("iai_mcp.daemon.write_event", return_value=None):
        task = asyncio.create_task(
            daemon_mod._hippea_cascade_loop(
                store=None,
                shutdown=shutdown,
                state_ref=state,
                _clock=lambda: 1000.0,
            )
        )
        await asyncio.sleep(0.08)
        shutdown.set()
        await asyncio.wait_for(task, timeout=1.0)

    assert len(calls) == 1
    assert state["hippea_cascade_request"] == {"pending": False}
