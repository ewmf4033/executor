"""
DO NOT RUN — known-broken Phase 4 test harness, preserved as a negative example.

This script reproduces the Phase 4 failure mode that Phase 4.5 fixed: the
strategy emits INTENT_EMITTED events but nothing is subscribed to consume
them through risk / fill / attribution. Running this produces hundreds of
intent events with no downstream pipeline activity — the exact silent-gap
bug the self-check was built to prevent from shipping again.

For actual paper runs, use the daemon:
  PAPER_MODE=true python -m executor --daemon

Entry point: executor/core/daemon.py :: DaemonService / run_daemon.

History: moved to scripts/regression/ in Phase 4.7 (Review 2 Q9) so the
file isn't accidentally picked up by operators looking for "burn-in" tools
in scripts/.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import signal
import sys
import time
from decimal import Decimal
from pathlib import Path

# Make sure we run as a module-style import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from executor.attribution.tracker import AttributionTracker  # noqa: E402
from executor.audit.writer import AuditWriter  # noqa: E402
from executor.core.event_bus import EventBus  # noqa: E402
from executor.core.events import Event, EventType, Source  # noqa: E402
from executor.core.logging import configure, get_logger  # noqa: E402
from executor.kill.manager import KillManager  # noqa: E402
from executor.kill.state import KillStateStore  # noqa: E402
from executor.strategies.yes_no_cross.strategy import (  # noqa: E402
    CrossPair,
    YESNOCrossDetect,
)


log = get_logger("scripts.burn_in_yes_no_cross")


async def quote_feeder(strategy: YESNOCrossDetect, *, stop: asyncio.Event) -> None:
    """
    Random-walk YES on kalshi and NO on polymarket so that a small fraction
    of the time YES.ask + NO.ask < 0.98, triggering basket emission.
    """
    rng = random.Random(0xC0FFEE)
    yes = Decimal("0.50")
    no_ = Decimal("0.50")
    while not stop.is_set():
        # symmetric jitter
        yes += Decimal(str(rng.uniform(-0.02, 0.02)))
        no_ += Decimal(str(rng.uniform(-0.02, 0.02)))
        # clamp to [0.01, 0.99]
        yes = max(Decimal("0.01"), min(Decimal("0.99"), yes))
        no_ = max(Decimal("0.01"), min(Decimal("0.99"), no_))
        strategy.accept_quote(
            venue="kalshi", market_id="K1", outcome_id="YES",
            best_ask=yes, mid=yes,
        )
        strategy.accept_quote(
            venue="polymarket", market_id="P1", outcome_id="NO",
            best_ask=no_, mid=no_,
        )
        await asyncio.sleep(0.5)


async def emit_loop(strategy: YESNOCrossDetect, *, stop: asyncio.Event) -> int:
    n = 0
    while not stop.is_set():
        try:
            emitted = await strategy.attempt_emit()
            n += len(emitted)
        except Exception as exc:  # pragma: no cover
            log.error("burn_in.emit.error", error=str(exc))
            return -1
        await asyncio.sleep(1.0)
    return n


async def amain(seconds: int, audit_dir: Path) -> int:
    configure()
    os.environ.setdefault("PAPER_MODE", "true")

    bus = EventBus()
    audit = AuditWriter(audit_dir)
    await audit.start()
    sub = await bus.subscribe("audit", on_event=audit.on_event)
    await bus.start()

    kstore = KillStateStore(audit_dir / "kill_state.sqlite")
    km = KillManager(store=kstore, publish=bus.publish)
    tracker = AttributionTracker(
        db_path=audit_dir / "attribution.sqlite",
        exit_horizon_sec=300,
        publish=bus.publish,
    )

    pair = CrossPair(
        yes_venue="kalshi", yes_market_id="K1",
        no_venue="polymarket", no_market_id="P1",
    )
    strategy = YESNOCrossDetect(pairs=[pair], emit_cooldown_sec=2.0)
    strategy.attach(bus.publish)

    await bus.publish(
        Event.make(
            EventType.EXECUTOR_LIFECYCLE,
            source=Source.EXECUTOR,
            payload={"state": "BURN_IN_STARTED", "seconds": seconds, "paper": True},
        )
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _stop() -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    feeder = asyncio.create_task(quote_feeder(strategy, stop=stop))
    emitter = asyncio.create_task(emit_loop(strategy, stop=stop))

    deadline = time.monotonic() + seconds
    try:
        while not stop.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(1.0)
        stop.set()
        await feeder
        n_emitted = await emitter
        log.info("burn_in.done", seconds=seconds, intents_emitted=n_emitted)
        await bus.publish(
            Event.make(
                EventType.EXECUTOR_LIFECYCLE,
                source=Source.EXECUTOR,
                payload={
                    "state": "BURN_IN_COMPLETE",
                    "intents_emitted": n_emitted,
                },
            )
        )
        rc = 0 if n_emitted >= 0 else 1
    finally:
        await bus.stop()
        await audit.stop()
        kstore.close()
        tracker.close()

    return rc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=int, default=600, help="burn-in duration")
    p.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("/root/executor/audit-logs/phase4_burn_in"),
    )
    args = p.parse_args()
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    sys.exit(asyncio.run(amain(args.seconds, args.audit_dir)))


if __name__ == "__main__":
    main()
