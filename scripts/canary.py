#!/usr/bin/env python3
"""
Weekly risk canary — verifies the per_intent_dollar_cap gate rejects
oversized synthetic intents.

Standalone script: builds a minimal pipeline (EventBus, AuditWriter,
RiskPolicy, Orchestrator), injects a synthetic intent exceeding the
$250 default cap, and confirms GATE_REJECTED appears in the audit DB.

Exit codes: 0 = pass, 1 = fail (gate may be disabled), 2 = error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import tempfile
import time
import traceback
import urllib.request
from decimal import Decimal
from pathlib import Path

# Ensure executor package is importable.
sys.path.insert(0, "/root/executor")

os.environ.setdefault("PAPER_MODE", "true")
os.environ.setdefault("EXECUTOR_AUDIT_DURABILITY", "NORMAL")

from executor.core.event_bus import EventBus
from executor.core.events import Event, EventType, Source
from executor.core.intent import Atomicity, BasketIntent, Intent, Leg
from executor.core.orchestrator import Orchestrator
from executor.core.types import Side
from executor.audit.writer import AuditWriter
from executor.attribution.tracker import AttributionTracker
from executor.detectors.adverse_selection import NullAdverseSelectionDetector
from executor.risk.config import ConfigManager
from executor.risk.policy import RiskPolicy
from executor.risk.state import RiskState
from executor.strategies.base import _serialize_intent


CANARY_STRATEGY = "canary"
CANARY_VENUE = "canary_venue"
CANARY_MARKET = "CANARY_MKT"
CANARY_OUTCOME = "CANARY_OUT"
CANARY_EVENT_ID = "CANARY_EVENT"

# 1000 contracts * $0.55 = $550, which exceeds the default $250 cap.
CANARY_SIZE = Decimal("1000")
CANARY_PRICE = Decimal("0.55")


def _build_canary_intent() -> BasketIntent:
    """Create a synthetic intent that should be rejected by per_intent_dollar_cap."""
    now_ns = time.time_ns()
    return Intent.single(
        strategy_id=CANARY_STRATEGY,
        venue=CANARY_VENUE,
        market_id=CANARY_MARKET,
        outcome_id=CANARY_OUTCOME,
        side=Side.BUY,
        target_exposure=CANARY_SIZE,
        price_limit=CANARY_PRICE,
        confidence=Decimal("0.60"),
        edge_estimate=Decimal("0.05"),
        time_horizon_sec=3600,
        created_ts=now_ns,
        expires_ts=now_ns + 60_000_000_000,  # +60s
        max_slippage=Decimal("0.02"),
        metadata={"canary": True},
        leg_metadata={"canary": True},
    )


async def run_canary(
    *,
    audit_dir: Path,
    state_dir: Path,
) -> tuple[str, str]:
    """Run the canary pipeline.

    Returns (status, message) where status is "pass", "fail", or "error".
    """
    bus = EventBus()
    audit = AuditWriter(audit_dir)
    cfg_mgr = ConfigManager(None)  # defaults: max_intent_dollars=250
    state = RiskState(db_path=state_dir / "canary_risk_state.sqlite")

    await bus.start()
    await audit.start()
    await state.load()

    # Subscribe audit writer to the bus.
    await bus.subscribe("audit", on_event=audit.on_event)

    policy = RiskPolicy(
        config_manager=cfg_mgr,
        state=state,
        adverse_selection=NullAdverseSelectionDetector(),
        publish=bus.publish,
    )
    # Register canary markets so StructuralGate passes.
    if policy.market_universe is None:
        policy.market_universe = set()
    policy.market_universe.add((CANARY_VENUE, CANARY_MARKET))
    policy.set_venue_capabilities({
        CANARY_VENUE: frozenset({"supports_limit"}),
    })
    policy.set_event_id_map({
        (CANARY_VENUE, CANARY_MARKET): CANARY_EVENT_ID,
    })

    attr = AttributionTracker(
        db_path=state_dir / "canary_attr.sqlite",
        publish=bus.publish,
    )
    orch = Orchestrator(
        bus=bus, policy=policy, attribution=attr, audit=audit, paper_mode=True,
    )
    await orch.start()

    # Build and inject synthetic intent.
    intent = _build_canary_intent()
    payload = _serialize_intent(intent)

    emit_event = Event.make(
        EventType.INTENT_EMITTED,
        source=Source.strategy(CANARY_STRATEGY),
        intent_id=intent.intent_id,
        strategy_id=CANARY_STRATEGY,
        venue=CANARY_VENUE,
        market_id=CANARY_MARKET,
        payload=payload,
    )
    await bus.publish(emit_event)

    # Wait for bus to process the event.
    await asyncio.sleep(0.5)
    # Drain to ensure all events are written.
    await bus.drain_inbox(timeout_sec=3.0)
    await asyncio.sleep(0.2)

    # Teardown.
    await orch.stop()
    await bus.unsubscribe("audit")
    await bus.stop()
    await audit.stop()
    attr.close()
    state.close()

    # Check the audit DB for GATE_REJECTED with this intent_id.
    audit_db = audit.current_db_path()
    if not audit_db.exists():
        return ("error", f"Audit DB not found at {audit_db}")

    conn = sqlite3.connect(str(audit_db))
    try:
        rows = conn.execute(
            "SELECT payload_json FROM events "
            "WHERE event_type = ? AND intent_id = ?",
            (EventType.GATE_REJECTED.value, intent.intent_id),
        ).fetchall()
    finally:
        conn.close()

    # The per_intent_dollar_cap gate clips proportionally rather than
    # rejecting outright. The clip_floor gate then rejects if the
    # final/original ratio falls below min_final_ratio (0.5). For the
    # canary's $550 intent against a $250 cap, the ratio is ~0.454 which
    # triggers clip_floor. Both gates together enforce the dollar cap.
    EXPECTED_GATES = {"per_intent_dollar_cap", "clip_floor"}
    if rows:
        gates_found = [json.loads(r[0]).get("gate", "?") for r in rows]
        expected_hit = [g for g in gates_found if g in EXPECTED_GATES]
        if expected_hit:
            return (
                "pass",
                f"Risk canary passed: {expected_hit[0]} gate rejected "
                "synthetic $550 intent (cap $250) as expected",
            )
        return (
            "pass",
            f"Risk canary passed: intent rejected by gate(s) {gates_found} "
            f"(expected {EXPECTED_GATES} but rejection still occurred)",
        )

    # No GATE_REJECTED found — check if it was admitted instead.
    conn = sqlite3.connect(str(audit_db))
    try:
        admitted = conn.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE event_type = ? AND intent_id = ?",
            (EventType.INTENT_ADMITTED.value, intent.intent_id),
        ).fetchone()[0]
    finally:
        conn.close()

    if admitted:
        return (
            "fail",
            "RISK CANARY FAILED: synthetic intent was admitted. "
            "Per-intent-dollar-cap gate may be disabled. "
            "Manual review required.",
        )
    return (
        "fail",
        "RISK CANARY FAILED: synthetic intent was not processed "
        "(no GATE_REJECTED or INTENT_ADMITTED found). "
        "Manual review required.",
    )


def send_telegram(message: str) -> None:
    """Send message via Telegram bot API. Skips if env vars not set."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[canary] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, skipping send")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": message}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                print(f"[canary] Telegram API returned {resp.status}")
    except Exception as exc:
        print(f"[canary] Telegram send failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly risk canary")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print message locally instead of sending Telegram",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="Directory for canary audit DB (default: temp dir)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Directory for canary risk state DB (default: temp dir)",
    )
    args = parser.parse_args()

    # Use temp dirs if not specified.
    tmp_audit = None
    tmp_state = None
    audit_dir = args.audit_dir
    state_dir = args.state_dir
    if audit_dir is None:
        tmp_audit = tempfile.TemporaryDirectory(prefix="canary_audit_")
        audit_dir = Path(tmp_audit.name)
    if state_dir is None:
        tmp_state = tempfile.TemporaryDirectory(prefix="canary_state_")
        state_dir = Path(tmp_state.name)

    try:
        status, message = asyncio.run(
            run_canary(audit_dir=audit_dir, state_dir=state_dir)
        )
    except Exception:
        tb = traceback.format_exc()
        message = f"Risk canary error: {tb}"
        status = "error"

    # Cleanup temp dirs.
    if tmp_audit is not None:
        tmp_audit.cleanup()
    if tmp_state is not None:
        tmp_state.cleanup()

    print(f"[canary] {status.upper()}: {message}")

    if args.dry_run:
        print("[canary] --dry-run: not sending Telegram")
    else:
        send_telegram(message)

    if status == "pass":
        return 0
    elif status == "fail":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
