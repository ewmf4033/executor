"""
Phase 4.11 integration tests — safety-critical findings from Codex Reviews 8 & 9.

Items covered:
  1. Shared KillSwitch between KillManager and RiskPolicy (split-brain fix)
  2. Daily PnL wiring from attribution settlement to RiskState.daily_pnl
  3. Orchestrator post-admission kill recheck emits ORDER_CANCELLED_PRE_SEND
  4. Kill config plumbing — auto_resume_strike_limit + panic_cooldown_sec
  5. KillStateStore DB corruption → ns-timestamped backup + rebuild

Each test is self-contained and avoids starting the full DaemonService
unless absolutely necessary (keeps the 1GB RAM constraint happy).
"""
from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import time
from decimal import Decimal
from pathlib import Path

import pytest

from executor.attribution.tracker import AttributionTracker
from executor.core.daemon import DaemonService
from executor.core.event_bus import EventBus
from executor.core.events import Event, EventType
from executor.core.intent import Atomicity, BasketIntent, Leg
from executor.core.orchestrator import Orchestrator
from executor.core.types import Side
from executor.detectors.adverse_selection import NullAdverseSelectionDetector
from executor.kill.manager import KillManager
from executor.kill.state import KillMode, KillStateStore
from executor.risk.config import ConfigManager
from executor.risk.kill import KillSwitch
from executor.risk.policy import RiskPolicy
from executor.risk.state import RiskState


# asyncio mark is applied per-test below so the Item 5 sync corruption
# tests don't trigger pytest-asyncio warnings.


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_leg(
    *,
    venue: str = "kalshi",
    market_id: str = "MKT-1",
    outcome_id: str = "YES",
    side: Side = Side.BUY,
    size: str = "10",
    price: str = "0.55",
    edge: str = "0.03",
    leg_id: str | None = None,
) -> Leg:
    kwargs = dict(
        venue=venue,
        market_id=market_id,
        outcome_id=outcome_id,
        side=side,
        target_exposure=Decimal(size),
        price_limit=Decimal(price),
        confidence=Decimal("0.6"),
        edge_estimate=Decimal(edge),
        time_horizon_sec=120,
        required_capabilities=(),
    )
    if leg_id is not None:
        kwargs["leg_id"] = leg_id
    return Leg(**kwargs)


def _make_intent(
    *,
    strategy_id: str = "s1",
    legs: list[Leg] | None = None,
    expires_in_sec: int = 60,
) -> BasketIntent:
    now = time.time_ns()
    legs = legs or [_make_leg()]
    from executor.core.intent import Intent
    if len(legs) == 1:
        leg = legs[0]
        return Intent.single(
            strategy_id=strategy_id,
            venue=leg.venue,
            market_id=leg.market_id,
            outcome_id=leg.outcome_id,
            side=leg.side,
            target_exposure=leg.target_exposure,
            price_limit=leg.price_limit,
            confidence=leg.confidence,
            edge_estimate=leg.edge_estimate,
            time_horizon_sec=leg.time_horizon_sec,
            required_capabilities=leg.required_capabilities,
            created_ts=now,
            expires_ts=now + expires_in_sec * 1_000_000_000,
        )
    return Intent.basket(
        strategy_id=strategy_id,
        legs=legs,
        atomicity=Atomicity.INDEPENDENT,
        max_slippage_per_leg=Decimal("0.02"),
        basket_target_exposure=sum((l.target_exposure for l in legs), Decimal("0")),
        created_ts=now,
        expires_ts=now + expires_in_sec * 1_000_000_000,
    )


async def _wire_policy_with_bus(
    tmp_path: Path,
    *,
    kill_switch: KillSwitch | None = None,
) -> tuple[RiskPolicy, EventBus, list[Event], RiskState]:
    state = RiskState(db_path=tmp_path / "rstate.sqlite")
    await state.load()
    cfg_mgr = ConfigManager(path=None)
    policy = RiskPolicy(
        config_manager=cfg_mgr,
        state=state,
        adverse_selection=NullAdverseSelectionDetector(),
        kill_switch=kill_switch,
    )
    policy.set_venue_capabilities(
        {"kalshi": {"supports_limit", "supports_market"}}
    )
    policy.set_allow_universe_bootstrap(True)
    policy.set_event_id_map({("kalshi", "MKT-1"): "EVT-MKT-1"})
    bus = EventBus()
    await bus.start()
    received: list[Event] = []

    async def sink(e: Event) -> None:
        received.append(e)

    await bus.subscribe("test", on_event=sink)
    policy.set_publish(bus.publish)
    return policy, bus, received, state


# --------------------------------------------------------------------------
# Item 1: Shared KillSwitch — engage via manager → policy rejects
# --------------------------------------------------------------------------


async def test_item1_shared_kill_switch_rejects_new_intent(tmp_path: Path) -> None:
    shared = KillSwitch()
    policy, bus, received, state = await _wire_policy_with_bus(
        tmp_path, kill_switch=shared
    )
    try:
        kill_store = KillStateStore(tmp_path / "kill.sqlite")
        kill_mgr = KillManager(
            store=kill_store, kill_switch=shared, publish=bus.publish
        )
        # Sanity: both sides see the same object before engage.
        assert kill_mgr._kill_switch is policy.kill_switch

        await kill_mgr.engage(KillMode.HARD, "test shared instance")

        intent = _make_intent(strategy_id="s1")
        verdict = await policy.evaluate(intent)

        assert verdict.admitted is False
        # Gate 2 (kill_switch) fires first — not the post-loop recheck.
        assert verdict.reject_gate == "kill_switch"

        await asyncio.sleep(0.02)
        rejects = [
            e for e in received
            if e.event_type == EventType.GATE_REJECTED
            and e.payload.get("gate") == "kill_switch"
        ]
        assert len(rejects) == 1
        # gate_order is serialized as a string ("2" for the kill_switch gate).
        assert str(rejects[0].payload.get("gate_order")) == "2"

        kill_store.close()
    finally:
        await bus.stop()
        state.close()


# --------------------------------------------------------------------------
# Item 2: Daily PnL wiring — settle_due writes record_pnl,
# subsequent intent hits daily_loss gate
# --------------------------------------------------------------------------


async def _settle_one_fill(
    *,
    state: RiskState,
    tmp_path: Path,
    side: Side,
    fill_price: str,
    exit_price: str,
    size: str = "10",
    strategy_id: str = "test_strat",
    fill_id: str = "f1",
) -> AttributionTracker:
    """Helper: on_fill + settle_due with exit_horizon=0 so the fill is
    immediately due. Returns the tracker so callers can close() it."""
    tracker = AttributionTracker(
        db_path=tmp_path / f"attr-{fill_id}.sqlite",
        exit_horizon_sec=0,
        risk_state=state,
    )
    tracker.update_mid("kalshi", "m1", Decimal(exit_price))
    tracker.on_fill(
        fill_id=fill_id,
        order_id=f"o-{fill_id}",
        intent_id=f"i-{fill_id}",
        leg_id=f"l-{fill_id}",
        strategy_id=strategy_id,
        venue="kalshi",
        market_id="m1",
        side=side,
        size=Decimal(size),
        fill_price=Decimal(fill_price),
        fill_ts_ns=time.time_ns(),
        intent_price=Decimal(fill_price),
    )
    settled = await tracker.settle_due(
        now_ns=time.time_ns() + 1_000_000_000
    )
    assert len(settled) == 1
    return tracker


async def test_item2_settle_due_records_daily_pnl_losing_buy(tmp_path: Path) -> None:
    """Phase 4.11.2: losing BUY (fill 0.50, exit 0.40, size 10) → pnl -1.00."""
    state = RiskState(db_path=tmp_path / "rstate.sqlite")
    await state.load()
    try:
        tracker = await _settle_one_fill(
            state=state, tmp_path=tmp_path,
            side=Side.BUY, fill_price="0.50", exit_price="0.40",
        )
        pnl = state.daily_pnl("test_strat")
        assert pnl == Decimal("-1.00"), f"expected -1.00 (loss), got {pnl}"
        tracker.close()
    finally:
        state.close()


async def test_item2_settle_due_records_daily_pnl_winning_buy(tmp_path: Path) -> None:
    """Phase 4.11.2: winning BUY (fill 0.40, exit 0.50, size 10) → pnl +1.00."""
    state = RiskState(db_path=tmp_path / "rstate.sqlite")
    await state.load()
    try:
        tracker = await _settle_one_fill(
            state=state, tmp_path=tmp_path,
            side=Side.BUY, fill_price="0.40", exit_price="0.50",
        )
        pnl = state.daily_pnl("test_strat")
        assert pnl == Decimal("1.00"), f"expected +1.00 (gain), got {pnl}"
        tracker.close()
    finally:
        state.close()


async def test_item2_settle_due_records_daily_pnl_losing_sell(tmp_path: Path) -> None:
    """Phase 4.11.2: losing SELL (sold 0.40, mid → 0.50, size 10) → pnl -1.00.
    Shorted at 0.40; mid moved up so it costs more to cover."""
    state = RiskState(db_path=tmp_path / "rstate.sqlite")
    await state.load()
    try:
        tracker = await _settle_one_fill(
            state=state, tmp_path=tmp_path,
            side=Side.SELL, fill_price="0.40", exit_price="0.50",
        )
        pnl = state.daily_pnl("test_strat")
        assert pnl == Decimal("-1.00"), f"expected -1.00 (loss), got {pnl}"
        tracker.close()
    finally:
        state.close()


async def test_item2_settle_due_records_daily_pnl_winning_sell(tmp_path: Path) -> None:
    """Phase 4.11.2: winning SELL (sold 0.50, mid → 0.40, size 10) → pnl +1.00."""
    state = RiskState(db_path=tmp_path / "rstate.sqlite")
    await state.load()
    try:
        tracker = await _settle_one_fill(
            state=state, tmp_path=tmp_path,
            side=Side.SELL, fill_price="0.50", exit_price="0.40",
        )
        pnl = state.daily_pnl("test_strat")
        assert pnl == Decimal("1.00"), f"expected +1.00 (gain), got {pnl}"
        tracker.close()
    finally:
        state.close()


async def test_item2_daily_pnl_gate_rejects_after_accumulated_loss(
    tmp_path: Path,
) -> None:
    """End-to-end: settled losses drive daily_pnl negative through the REAL
    attribution plumbing (settle_due → record_pnl) — not a bypass via
    direct state.record_pnl — and a fresh intent hits gate_13 (daily_loss).

    Phase 4.11.2 (Codex Review 10): previously this test bypassed the
    attribution path, failing to validate end-to-end PnL accrual. Now we
    drive 30 losing BUY fills (each = $1.00 loss) through settle_due and
    assert daily_pnl crosses the default -$200 daily_loss threshold, then
    that a subsequent policy.evaluate() rejects with gate=daily_loss."""
    state = RiskState(db_path=tmp_path / "rstate.sqlite")
    await state.load()
    try:
        tracker = AttributionTracker(
            db_path=tmp_path / "attr.sqlite",
            exit_horizon_sec=0,
            risk_state=state,
        )
        # Each fill: BUY at 0.50, mid drops to 0.40, size 10 → pnl -1.00.
        # 30 fills × -1.00 = -$30.00. Need to exceed default daily_loss cap
        # of -$200, so do 250 fills → -$250.00.
        tracker.update_mid("kalshi", "m1", Decimal("0.40"))
        now = time.time_ns()
        for i in range(250):
            tracker.on_fill(
                fill_id=f"f{i}",
                order_id=f"o{i}",
                intent_id=f"i{i}",
                leg_id=f"l{i}",
                strategy_id="test_strat",
                venue="kalshi",
                market_id="m1",
                side=Side.BUY,
                size=Decimal("10"),
                fill_price=Decimal("0.50"),
                fill_ts_ns=now,
                intent_price=Decimal("0.50"),
            )
        settled = await tracker.settle_due(now_ns=now + 1_000_000_000)
        assert len(settled) == 250
        pnl = state.daily_pnl("test_strat")
        assert pnl == Decimal("-250.00"), f"expected -250.00, got {pnl}"

        # Now evaluate a fresh intent for the same strategy — gate_13
        # (daily_loss) must reject it.
        cfg_mgr = ConfigManager(path=None)
        policy = RiskPolicy(
            config_manager=cfg_mgr,
            state=state,
            adverse_selection=NullAdverseSelectionDetector(),
        )
        policy.set_venue_capabilities(
            {"kalshi": {"supports_limit", "supports_market"}}
        )
        policy.set_allow_universe_bootstrap(True)
        policy.set_event_id_map({("kalshi", "MKT-1"): "EVT-MKT-1"})

        intent = _make_intent(strategy_id="test_strat")
        verdict = await policy.evaluate(intent)
        assert verdict.admitted is False
        assert verdict.reject_gate == "daily_loss"

        tracker.close()
    finally:
        state.close()


# --------------------------------------------------------------------------
# Item 3: Orchestrator post-admission kill recheck
# --------------------------------------------------------------------------


async def test_item3_paper_fill_skips_when_killed(tmp_path: Path) -> None:
    shared = KillSwitch()
    policy, bus, received, state = await _wire_policy_with_bus(
        tmp_path, kill_switch=shared
    )
    try:
        orch = Orchestrator(
            bus=bus,
            policy=policy,
            attribution=None,
            audit=None,
            paper_mode=True,
        )

        intent = _make_intent(strategy_id="s1")
        verdict = await policy.evaluate(intent)
        assert verdict.admitted is True

        # Engage kill between admission and fill.
        from executor.risk.kill import KillScope
        shared.engage(KillScope.GLOBAL, (), "test post-admission kill")

        received.clear()
        await orch._paper_fill(verdict.intent)
        await asyncio.sleep(0.02)

        types = [e.event_type for e in received]
        assert EventType.ORDER_PLACED not in types
        assert EventType.FILL not in types
        assert EventType.ORDER_CANCELLED_PRE_SEND in types

        pre_send = next(
            e for e in received
            if e.event_type == EventType.ORDER_CANCELLED_PRE_SEND
        )
        payload = pre_send.payload
        assert payload["kill_reason"].startswith("GLOBAL:")
        assert payload["intent_id"] == intent.intent_id
        assert payload["strategy_id"] == "s1"
        assert payload["n_legs_skipped"] == len(intent.legs)
    finally:
        await bus.stop()
        state.close()


# --------------------------------------------------------------------------
# Item 4: Kill config plumbing — custom YAML values reach KillManager
# --------------------------------------------------------------------------


async def test_item4_kill_config_reaches_manager(tmp_path: Path) -> None:
    yaml_path = tmp_path / "risk.yaml"
    yaml_path.write_text(
        "kill_switch:\n"
        "  auto_resume_strike_limit: 5\n"
        "  panic_cooldown_sec: 60\n"
    )
    audit_dir = tmp_path / "audit"
    os.environ["PAPER_MODE"] = "true"
    os.environ["EXECUTOR_PAPER_MODE_NO_ORDERBOOK"] = "true"
    svc = DaemonService(
        audit_dir=audit_dir,
        risk_yaml=yaml_path,
        risk_state_db=tmp_path / "rstate.sqlite",
        kill_db=tmp_path / "kill.sqlite",
        attribution_db=tmp_path / "attr.sqlite",
        telemetry_port=0,
        enable_quote_feeder=False,
        enable_self_check=False,
    )
    try:
        await svc.start()
        assert svc.kill_mgr is not None
        assert svc.kill_mgr._auto_resume_strike_limit == 5
        assert svc.kill_mgr._panic_cooldown_sec == 60
    finally:
        await svc.stop()


async def test_item4_config_reload_updates_manager(tmp_path: Path) -> None:
    yaml_path = tmp_path / "risk.yaml"
    yaml_path.write_text(
        "kill_switch:\n"
        "  auto_resume_strike_limit: 3\n"
        "  panic_cooldown_sec: 300\n"
    )
    audit_dir = tmp_path / "audit"
    os.environ["PAPER_MODE"] = "true"
    os.environ["EXECUTOR_PAPER_MODE_NO_ORDERBOOK"] = "true"
    svc = DaemonService(
        audit_dir=audit_dir,
        risk_yaml=yaml_path,
        risk_state_db=tmp_path / "rstate.sqlite",
        kill_db=tmp_path / "kill.sqlite",
        attribution_db=tmp_path / "attr.sqlite",
        telemetry_port=0,
        enable_quote_feeder=False,
        enable_self_check=False,
    )
    try:
        await svc.start()
        assert svc.kill_mgr._auto_resume_strike_limit == 3
        assert svc.kill_mgr._panic_cooldown_sec == 300

        yaml_path.write_text(
            "kill_switch:\n"
            "  auto_resume_strike_limit: 7\n"
            "  panic_cooldown_sec: 900\n"
        )
        await svc.config_mgr.reload()
        assert svc.kill_mgr._auto_resume_strike_limit == 7
        assert svc.kill_mgr._panic_cooldown_sec == 900
    finally:
        await svc.stop()


# --------------------------------------------------------------------------
# Item 5: KillStateStore DB corruption handling
# --------------------------------------------------------------------------


def test_item5_corrupt_db_renamed_and_rebuilt(tmp_path: Path) -> None:
    db_path = tmp_path / "kill_state.sqlite"
    db_path.write_bytes(b"this is not a SQLite database at all")

    store = KillStateStore(db_path)

    assert store.rebuilt_from_corruption is True
    snap = store.load()
    assert snap.mode == KillMode.NONE

    pattern = re.compile(r"^kill_state\.sqlite\.corrupt-\d{19}$")
    siblings = [p.name for p in tmp_path.iterdir()]
    matches = [n for n in siblings if pattern.match(n)]
    assert len(matches) == 1, f"expected one backup, got: {siblings}"

    store.close()


def test_item5_successive_corruption_unique_backups(tmp_path: Path) -> None:
    db_path = tmp_path / "kill_state.sqlite"

    db_path.write_bytes(b"garbage 1")
    store_a = KillStateStore(db_path)
    store_a.close()

    # Second corruption: overwrite the now-fresh DB with garbage again.
    db_path.write_bytes(b"garbage 2 (different)")
    store_b = KillStateStore(db_path)
    store_b.close()

    pattern = re.compile(r"^kill_state\.sqlite\.corrupt-(\d{19})$")
    matches = [
        pattern.match(p.name) for p in tmp_path.iterdir()
    ]
    ns_suffixes = sorted(int(m.group(1)) for m in matches if m)
    assert len(ns_suffixes) == 2, (
        f"expected two distinct backups, got: {list(tmp_path.iterdir())}"
    )
    assert ns_suffixes[0] != ns_suffixes[1]


def test_item5_clean_db_does_not_set_flag(tmp_path: Path) -> None:
    db_path = tmp_path / "kill_state.sqlite"
    store = KillStateStore(db_path)
    assert store.rebuilt_from_corruption is False
    store.close()

    # Reopening a clean existing DB should also not flag.
    store2 = KillStateStore(db_path)
    assert store2.rebuilt_from_corruption is False
    store2.close()
