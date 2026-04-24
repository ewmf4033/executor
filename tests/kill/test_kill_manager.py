"""KillManager — engage/resume, basket orphan, strike circuit."""
from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from pathlib import Path

import pytest

from executor.core.events import Event, EventType
from executor.core.intent import Atomicity, Intent, Leg
from executor.core.types import Side
from executor.kill.manager import KillManager
from executor.kill.state import KillMode, KillStateStore


class _FakeAdapter:
    def __init__(self, *, fail_ids: set[str] | None = None) -> None:
        self.cancelled: list[str] = []
        self._fail_ids = fail_ids or set()

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self._fail_ids:
            return False
        self.cancelled.append(order_id)
        return True


def _make_basket(*, atomicity: Atomicity, n: int = 2) -> Intent:
    legs = tuple(
        Leg(
            venue=("kalshi" if i == 0 else "polymarket"),
            market_id=f"M{i}",
            outcome_id=("YES" if i == 0 else "NO"),
            side=Side.BUY,
            target_exposure=Decimal("1"),
            price_limit=Decimal("0.5"),
            confidence=Decimal("0.8"),
            edge_estimate=Decimal("0.05"),
            time_horizon_sec=60,
        )
        for i in range(n)
    )
    return Intent.basket(
        strategy_id="t",
        legs=legs,
        atomicity=atomicity,
        max_slippage_per_leg=Decimal("0.02"),
        basket_target_exposure=Decimal("1"),
        created_ts=time.time_ns(),
        expires_ts=time.time_ns() + 30_000_000_000,
    )


def _mgr(tmp_path: Path) -> tuple[KillManager, list[Event]]:
    store = KillStateStore(tmp_path / "kill.sqlite")
    captured: list[Event] = []

    async def pub(ev: Event) -> None:
        captured.append(ev)

    m = KillManager(store=store, publish=pub, panic_cooldown_sec=1)
    return m, captured


def test_engage_soft_blocks_new_intents(tmp_path: Path) -> None:
    m, events = _mgr(tmp_path)
    asyncio.run(m.engage(KillMode.SOFT, "manual"))
    killed, why = m.is_killed()
    assert killed is True
    assert "SOFT" in why and "manual" in why
    types = [e.event_type for e in events]
    assert EventType.KILL_STATE_CHANGED in types
    assert EventType.KILL_SWITCH_TOGGLED in types


def test_engage_requires_reason(tmp_path: Path) -> None:
    m, _ = _mgr(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(m.engage(KillMode.SOFT, ""))


def test_engage_none_rejected(tmp_path: Path) -> None:
    m, _ = _mgr(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(m.engage(KillMode.NONE, "x"))


def test_resume_clears_state(tmp_path: Path) -> None:
    m, _ = _mgr(tmp_path)
    asyncio.run(m.engage(KillMode.SOFT, "x"))
    ok, why = asyncio.run(m.resume())
    assert ok is True
    assert why == ""
    killed, _ = m.is_killed()
    assert killed is False


def test_panic_blocks_resume_until_cooldown(tmp_path: Path) -> None:
    m, _ = _mgr(tmp_path)
    asyncio.run(m.engage(KillMode.HARD, "boom", panic=True))
    ok, why = asyncio.run(m.resume())
    assert ok is False
    # manual_only set; even after cooldown expires we still need force.
    assert "panic" in why.lower() or "manual" in why.lower()
    # force=True clears it.
    ok2, _ = asyncio.run(m.resume(force=True))
    assert ok2 is True


def test_strike_circuit_breaker(tmp_path: Path) -> None:
    m, _ = _mgr(tmp_path)
    for _ in range(2):
        assert m.mark_resume_health(healthy=False) is False
    # Third strike pins manual_only.
    assert m.mark_resume_health(healthy=False) is True
    assert m.snapshot().manual_only is True
    # Healthy clears strike count but not the manual_only flag.
    m.mark_resume_health(healthy=True)
    assert m.snapshot().resume_strikes == 0
    assert m.snapshot().manual_only is True
    # Operator override.
    m.clear_manual_only()
    assert m.snapshot().manual_only is False


def test_hard_cancels_open_orders(tmp_path: Path) -> None:
    m, events = _mgr(tmp_path)
    adapter_k = _FakeAdapter()
    adapter_p = _FakeAdapter()
    m.register_adapter("kalshi", adapter_k)
    m.register_adapter("polymarket", adapter_p)

    intent = _make_basket(atomicity=Atomicity.INDEPENDENT)
    leg_a, leg_b = intent.legs
    m.record_basket(intent, open_orders={leg_a.leg_id: ["o1"], leg_b.leg_id: ["o2"]})

    asyncio.run(m.engage(KillMode.HARD, "halt"))
    assert adapter_k.cancelled == ["o1"]
    assert adapter_p.cancelled == ["o2"]
    types = [e.event_type for e in events]
    # No filled legs -> BASKET_CANCELLED
    assert EventType.BASKET_CANCELLED in types
    assert EventType.BASKET_ORPHAN not in types


def test_hard_emits_basket_orphan_for_partial_all_or_none(tmp_path: Path) -> None:
    m, events = _mgr(tmp_path)
    adapter_k = _FakeAdapter()
    adapter_p = _FakeAdapter()
    m.register_adapter("kalshi", adapter_k)
    m.register_adapter("polymarket", adapter_p)

    intent = _make_basket(atomicity=Atomicity.ALL_OR_NONE)
    leg_a, leg_b = intent.legs
    m.record_basket(intent, open_orders={leg_a.leg_id: ["oA"], leg_b.leg_id: ["oB"]})
    # Leg A filled; leg B still open.
    m.mark_leg_filled(intent.intent_id, leg_a.leg_id)

    asyncio.run(m.engage(KillMode.HARD, "halt"))
    types = [e.event_type for e in events]
    assert EventType.BASKET_ORPHAN in types
    # Only the still-open order gets cancelled.
    assert adapter_p.cancelled == ["oB"]


def test_kill_command_received_event(tmp_path: Path) -> None:
    m, events = _mgr(tmp_path)
    asyncio.run(m.emit_command_received("kill", "soft halt", chat_id="42"))
    assert any(e.event_type == EventType.KILL_COMMAND_RECEIVED for e in events)
    [ev] = [e for e in events if e.event_type == EventType.KILL_COMMAND_RECEIVED]
    assert ev.payload["command"] == "kill"
    assert ev.payload["args"] == "soft halt"
    assert ev.payload["chat_id"] == "42"


# ---------------------------------------------------------------------------
# Phase 4.14d — Codex review fix: monotonic kill severity.
#
# Severity ordering: NONE < SOFT < HARD. An engage call whose mode is
# strictly weaker than the current mode must not downgrade state, and a
# non-panic engage must not clear an existing panic flag. The watchdog
# (phase 4.14c) calls ``engage(SOFT, ...)``; before this fix that call
# could silently overwrite an operator-issued HARD/PANIC. Only
# ``resume()`` may reduce kill state.
# ---------------------------------------------------------------------------


def test_kill_manager_engage_soft_does_not_downgrade_hard(tmp_path: Path) -> None:
    m, events = _mgr(tmp_path)
    # Operator engages HARD (no panic).
    snap = asyncio.run(m.engage(KillMode.HARD, "operator-hard"))
    assert snap.mode is KillMode.HARD
    assert snap.reason == "operator-hard"
    hard_engaged_at = snap.engaged_ts_ns

    # Watchdog (or any later caller) requests SOFT — must not downgrade.
    snap2 = asyncio.run(
        m.engage(KillMode.SOFT, "watchdog-soft", source="telegram_watchdog")
    )
    assert snap2.mode is KillMode.HARD, "SOFT must not downgrade HARD"
    assert snap2.reason == "operator-hard", "stricter prior reason preserved"
    # engaged_ts_ns is not refreshed on no-op engages (no severity raise).
    assert snap2.engaged_ts_ns == hard_engaged_at
    # A state_changed event is still emitted for audit visibility, but
    # from_mode == to_mode == HARD (no-op signal). Phase 4.14d: the
    # payload now carries explicit effective/skipped_weaker/requested_mode
    # so consumers don't have to infer the no-op from equal modes.
    state_changes = [
        e for e in events if e.event_type == EventType.KILL_STATE_CHANGED
    ]
    assert len(state_changes) >= 2
    last = state_changes[-1].payload
    assert last["from_mode"] == "HARD"
    assert last["to_mode"] == "HARD"
    assert last["requested_mode"] == "SOFT"
    assert last["effective"] is False
    assert last["skipped_weaker"] is True


def test_kill_manager_hard_can_upgrade_to_panic(tmp_path: Path) -> None:
    """HARD → HARD+PANIC upgrade: same-severity engage with panic=True
    must still set panic/manual_only/panic_until_ns even though the
    mode itself doesn't change. This exercises the ``sets_new_panic``
    branch of the effective-engage gate."""
    m, events = _mgr(tmp_path)
    # Start HARD without panic.
    snap = asyncio.run(m.engage(KillMode.HARD, "initial-hard"))
    assert snap.mode is KillMode.HARD
    assert snap.panic is False
    assert snap.manual_only is False
    assert snap.panic_until_ns == 0

    # Upgrade to panic — same mode, but adds panic flag.
    snap2 = asyncio.run(
        m.engage(KillMode.HARD, "now-panic", panic=True)
    )
    assert snap2.mode is KillMode.HARD
    assert snap2.panic is True
    assert snap2.manual_only is True
    assert snap2.panic_until_ns > 0
    # Phase 4.14e: reason is now updated on HARD → HARD+PANIC so
    # the operator-supplied panic reason survives into
    # ``kill_status`` and the audit trail. ``engaged_ts_ns`` is
    # still NOT refreshed (no severity raise), preserving the
    # original HARD engagement time for forensics.
    assert snap2.reason == "now-panic"
    state_changes = [
        e for e in events if e.event_type == EventType.KILL_STATE_CHANGED
    ]
    assert state_changes[-1].payload["panic"] is True
    assert state_changes[-1].payload["to_mode"] == "HARD"
    assert state_changes[-1].payload["from_mode"] == "HARD"
    assert state_changes[-1].payload["reason"] == "now-panic"


def test_kill_manager_engage_soft_does_not_clear_panic(tmp_path: Path) -> None:
    m, events = _mgr(tmp_path)
    # Operator engages HARD with panic.
    snap = asyncio.run(m.engage(KillMode.HARD, "operator-panic", panic=True))
    assert snap.mode is KillMode.HARD
    assert snap.panic is True
    assert snap.manual_only is True

    # Watchdog requests SOFT without panic — must not clear panic.
    snap2 = asyncio.run(
        m.engage(KillMode.SOFT, "watchdog-soft", source="telegram_watchdog")
    )
    assert snap2.mode is KillMode.HARD
    assert snap2.panic is True, "panic flag must survive weaker engage"
    assert snap2.manual_only is True

    # A non-panic HARD engage must also not clear an existing panic.
    snap3 = asyncio.run(
        m.engage(KillMode.HARD, "second-hard", panic=False)
    )
    assert snap3.panic is True


# ---------------------------------------------------------------------------
# Phase 4.14e — HARD → PANIC upgrade: reason preservation, panic bookkeeping,
# and engaged_ts_ns stability.
#
# The ultrareview Run 2 flagged H11: the operator-supplied panic reason was
# dropped on HARD → PANIC because ``final_reason`` only updated on a
# severity raise. HARD → HARD+PANIC raises no severity (same mode) but is
# still an effective engage (``sets_new_panic=True``). We now update
# ``reason`` on any effective engage while keeping ``engaged_ts_ns`` frozen
# on non-severity-raising transitions so forensic ordering survives.
# ---------------------------------------------------------------------------


def test_hard_to_panic_updates_reason(tmp_path: Path) -> None:
    m, _ = _mgr(tmp_path)
    asyncio.run(m.engage(KillMode.HARD, "initial-hard"))
    snap = asyncio.run(
        m.engage(KillMode.HARD, "operator panic after incident", panic=True)
    )
    assert snap.mode is KillMode.HARD
    assert snap.panic is True
    # The operator's panic reason must survive into the persisted snapshot.
    assert snap.reason == "operator panic after incident"


def test_hard_to_panic_preserves_original_engaged_ts_ns(tmp_path: Path) -> None:
    m, _ = _mgr(tmp_path)
    snap_first = asyncio.run(m.engage(KillMode.HARD, "initial-hard"))
    original_ts = snap_first.engaged_ts_ns
    assert original_ts > 0
    # Let the wall clock advance a hair so a refreshed timestamp would differ.
    time.sleep(0.002)
    snap_panic = asyncio.run(
        m.engage(KillMode.HARD, "panic", panic=True)
    )
    # engaged_ts_ns refreshes only on severity raises (NONE→SOFT, SOFT→HARD,
    # NONE→HARD), NOT on a panic-only upgrade at the same mode.
    assert snap_panic.engaged_ts_ns == original_ts


def test_hard_to_panic_refreshes_panic_until_and_manual_only(tmp_path: Path) -> None:
    m, _ = _mgr(tmp_path)
    asyncio.run(m.engage(KillMode.HARD, "initial-hard"))
    snap_pre = m.snapshot()
    assert snap_pre.panic is False
    assert snap_pre.panic_until_ns == 0
    assert snap_pre.manual_only is False

    snap = asyncio.run(m.engage(KillMode.HARD, "panic", panic=True))
    assert snap.panic is True
    # panic_until_ns is refreshed to now + cooldown (cooldown=1s in _mgr).
    assert snap.panic_until_ns > 0
    # manual_only is set on any panic engage.
    assert snap.manual_only is True


def test_state_persists_across_restart(tmp_path: Path) -> None:
    db = tmp_path / "k.sqlite"

    async def first() -> None:
        store = KillStateStore(db)
        m = KillManager(store=store, publish=None)
        await m.engage(KillMode.SOFT, "carry-over")
        store.close()

    asyncio.run(first())

    store2 = KillStateStore(db)
    m2 = KillManager(store=store2, publish=None)
    assert m2.mode is KillMode.SOFT
    killed, _ = m2.is_killed()
    assert killed is True
    store2.close()
