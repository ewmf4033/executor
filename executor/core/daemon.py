"""
DaemonService — long-running paper-mode executor wiring.

Fills the gap Phase 4's burn-in exposed: the strategy emits intents but
nothing consumed them. DaemonService wires the full pipeline:

  EventBus
  ├── AuditWriter                      (subscribed: every event)
  ├── TelemetryServer                  (subscribed: /pipeline_stats counts)
  ├── Orchestrator                     (subscribed: INTENT_EMITTED)
  │     └── RiskPolicy.evaluate()
  │           └── emits GATE_* / INTENT_ADMITTED
  │     └── synthetic paper FILL
  │           └── AttributionTracker.on_fill
  ├── RiskState                        (backing store)
  ├── ConfigManager                    (YAML + SIGHUP reload)
  ├── KillStateStore + KillManager
  ├── Strategy: YESNOCrossDetect       (emit() -> bus)
  └── Synthetic QuoteFeeder            (drives the strategy in paper daemon)

PAPER_MODE is hard-locked: the CLI pins it before we get here; this module
asserts it and refuses to run if the env was tampered with.
"""
from __future__ import annotations

import asyncio
import os
import random
import signal
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..attribution.tracker import AttributionTracker
from ..audit.writer import AuditWriter
from ..kill.manager import KillManager
from ..kill.state import KillStateStore
from ..risk.config import ConfigManager
from ..risk.policy import RiskPolicy
from ..risk.state import RiskState
from ..strategies.yes_no_cross.strategy import CrossPair, YESNOCrossDetect
from .event_bus import EventBus
from .events import Event, EventType, Source
from .logging import configure, get_logger
from .orchestrator import Orchestrator
from .self_check import run_self_check
from .telemetry import TelemetryServer


log = get_logger("executor.daemon")


DEFAULT_AUDIT_DIR = "/root/executor/audit-logs/paper_live"
DEFAULT_RISK_YAML = "/root/executor/config/risk.yaml"
DEFAULT_RISK_STATE_DB = "/root/executor/state/risk_state.sqlite"
DEFAULT_KILL_DB = "/root/executor/state/kill_state.sqlite"
DEFAULT_ATTRIBUTION_DB = "/root/executor/state/attribution.sqlite"

SELF_CHECK_DEADLINE_SEC = 60.0  # must complete within this window of startup


class DaemonService:
    def __init__(
        self,
        *,
        audit_dir: str | os.PathLike[str] | None = None,
        risk_yaml: str | os.PathLike[str] | None = None,
        risk_state_db: str | os.PathLike[str] | None = None,
        kill_db: str | os.PathLike[str] | None = None,
        attribution_db: str | os.PathLike[str] | None = None,
        telemetry_port: int = 9879,
        enable_quote_feeder: bool = True,
        enable_self_check: bool = True,
    ) -> None:
        # Hard-lock paper mode.
        if os.environ.get("PAPER_MODE", "true").lower() in ("0", "false", "no", "off"):
            raise RuntimeError(
                "DaemonService requires PAPER_MODE=true; --live should go through cli.resolve_mode"
            )
        self.audit_dir = Path(audit_dir or os.environ.get("EXECUTOR_AUDIT_DIR", DEFAULT_AUDIT_DIR))
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.risk_yaml = Path(risk_yaml or DEFAULT_RISK_YAML)
        self.risk_state_db = Path(risk_state_db or DEFAULT_RISK_STATE_DB)
        self.kill_db = Path(kill_db or DEFAULT_KILL_DB)
        self.attribution_db = Path(attribution_db or DEFAULT_ATTRIBUTION_DB)
        self.telemetry_port = telemetry_port
        self.enable_quote_feeder = enable_quote_feeder
        self.enable_self_check = enable_self_check

        self.bus = EventBus()
        self.audit = AuditWriter(self.audit_dir)
        self.risk_state: RiskState | None = None
        self.config_mgr: ConfigManager | None = None
        self.kill_store: KillStateStore | None = None
        self.kill_mgr: KillManager | None = None
        self.policy: RiskPolicy | None = None
        self.attribution: AttributionTracker | None = None
        self.telemetry: TelemetryServer | None = None
        self.orchestrator: Orchestrator | None = None
        self.strategy: YESNOCrossDetect | None = None

        self._stop_event = asyncio.Event()
        self._bg_tasks: list[asyncio.Task[Any]] = []
        self._started = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(self) -> None:
        configure()
        # Ensure state dirs exist.
        for p in (self.risk_state_db, self.kill_db, self.attribution_db):
            p.parent.mkdir(parents=True, exist_ok=True)

        # Audit first — we want every subsequent event captured.
        await self.audit.start()
        await self.bus.subscribe("audit", on_event=self.audit.on_event, queue_maxsize=10_000)

        # Telemetry — subscribes to everything; safe to start early.
        self.telemetry = TelemetryServer(port=self.telemetry_port, daemon_mode=True)
        await self.bus.subscribe("telemetry", on_event=self.telemetry.on_event, queue_maxsize=10_000)
        await self.telemetry.start()

        # Risk subsystem.
        self.config_mgr = ConfigManager(self.risk_yaml if self.risk_yaml.exists() else None)
        self.risk_state = RiskState(db_path=self.risk_state_db)
        await self.risk_state.load()

        self.kill_store = KillStateStore(self.kill_db)
        self.kill_mgr = KillManager(store=self.kill_store, publish=self.bus.publish)

        self.attribution = AttributionTracker(
            db_path=self.attribution_db,
            exit_horizon_sec=self.config_mgr.config.attribution.exit_horizon_sec,
            publish=self.bus.publish,
        )

        self.policy = RiskPolicy(
            config_manager=self.config_mgr,
            state=self.risk_state,
            publish=self.bus.publish,
        )
        self.policy.set_venue_capabilities(
            {"kalshi": frozenset({"supports_limit", "supports_market"}),
             "polymarket": frozenset({"supports_limit"}),
             "self_check_yes": frozenset({"supports_limit"}),
             "self_check_no": frozenset({"supports_limit"})}
        )

        # Bus must be pumping before orchestrator subscribes.
        await self.bus.start()

        # Orchestrator: INTENT_EMITTED → risk → paper FILL → attribution.
        self.orchestrator = Orchestrator(
            bus=self.bus,
            policy=self.policy,
            attribution=self.attribution,
            paper_mode=True,
        )
        await self.orchestrator.start()
        self.telemetry.set_orchestrator(self.orchestrator)

        # Strategy.
        pair = CrossPair(
            yes_venue="kalshi",
            yes_market_id="K1",
            no_venue="polymarket",
            no_market_id="P1",
        )
        self.strategy = YESNOCrossDetect(
            pairs=[pair],
            emit_cooldown_sec=2.0,
        )
        self.strategy.attach(self.bus.publish)

        await self.bus.publish(
            Event.make(
                EventType.EXECUTOR_LIFECYCLE,
                source=Source.EXECUTOR,
                payload={
                    "state": "DAEMON_STARTED",
                    "audit_dir": str(self.audit_dir),
                    "paper": True,
                    "telemetry_port": self.telemetry_port,
                },
            )
        )
        self._started = True
        log.info(
            "daemon.start",
            audit_dir=str(self.audit_dir),
            telemetry_port=self.telemetry_port,
        )

    # ------------------------------------------------------------------
    # Self-check
    # ------------------------------------------------------------------

    async def run_startup_self_check(self) -> dict[str, Any]:
        """Run the pipeline self-check; return the structured result dict."""
        assert self._started, "call start() first"
        try:
            result = await asyncio.wait_for(
                run_self_check(bus=self.bus, attribution=self.attribution),
                timeout=SELF_CHECK_DEADLINE_SEC,
            )
        except asyncio.TimeoutError:
            result = {
                "kind": "fail",
                "reason": f"self-check exceeded {SELF_CHECK_DEADLINE_SEC}s deadline",
                "stages_ms": {},
            }
            await self.bus.publish(
                Event.make(
                    EventType.SELF_CHECK_FAIL,
                    source=Source.EXECUTOR,
                    payload=result,
                )
            )
        return result

    # ------------------------------------------------------------------
    # Running forever
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        """
        Install signal handlers, spawn background tasks (quote feeder,
        strategy loop, attribution settlement), and block until signaled.
        """
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop_event.set)
            except NotImplementedError:
                pass

        if self.enable_quote_feeder:
            self._bg_tasks.append(
                asyncio.create_task(self._quote_feeder(), name="quote-feeder")
            )
        self._bg_tasks.append(
            asyncio.create_task(self._strategy_loop(), name="strategy-loop")
        )
        self._bg_tasks.append(
            asyncio.create_task(self._attribution_settle_loop(), name="attribution-settle")
        )
        await self._stop_event.wait()

    async def _quote_feeder(self) -> None:
        """Random-walk synthetic quotes so the strategy has something to detect."""
        assert self.strategy is not None
        rng = random.Random(0xBEEF)
        yes = Decimal("0.50")
        no_ = Decimal("0.50")
        try:
            while not self._stop_event.is_set():
                yes += Decimal(str(rng.uniform(-0.02, 0.02)))
                no_ += Decimal(str(rng.uniform(-0.02, 0.02)))
                yes = max(Decimal("0.01"), min(Decimal("0.99"), yes))
                no_ = max(Decimal("0.01"), min(Decimal("0.99"), no_))
                self.strategy.accept_quote(
                    venue="kalshi", market_id="K1", outcome_id="YES",
                    best_ask=yes, mid=yes,
                )
                self.strategy.accept_quote(
                    venue="polymarket", market_id="P1", outcome_id="NO",
                    best_ask=no_, mid=no_,
                )
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=0.5)
                    return
                except asyncio.TimeoutError:
                    pass
        except Exception as exc:  # pragma: no cover
            log.error("daemon.quote_feeder.crash", error=str(exc))

    async def _strategy_loop(self) -> None:
        assert self.strategy is not None
        try:
            while not self._stop_event.is_set():
                try:
                    await self.strategy.attempt_emit()
                except Exception as exc:
                    log.warning("daemon.strategy.emit_error", error=str(exc))
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                    return
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    async def _attribution_settle_loop(self) -> None:
        assert self.attribution is not None
        try:
            while not self._stop_event.is_set():
                try:
                    await self.attribution.settle_due()
                except Exception as exc:
                    log.warning("daemon.attribution.settle_error", error=str(exc))
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
                    return
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        self._stop_event.set()
        # Cancel background tasks.
        for t in self._bg_tasks:
            t.cancel()
        for t in self._bg_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._bg_tasks.clear()

        # Final lifecycle event before we tear down publishers.
        try:
            await self.bus.publish(
                Event.make(
                    EventType.EXECUTOR_LIFECYCLE,
                    source=Source.EXECUTOR,
                    payload={"state": "STOPPING"},
                )
            )
            await self.bus.publish(
                Event.make(
                    EventType.STATE_SAVED,
                    source=Source.EXECUTOR,
                    payload={
                        "audit_dir": str(self.audit_dir),
                        "risk_state_db": str(self.risk_state_db),
                        "kill_db": str(self.kill_db),
                        "attribution_db": str(self.attribution_db),
                    },
                )
            )
        except Exception as exc:  # pragma: no cover
            log.warning("daemon.final_events_failed", error=str(exc))

        if self.orchestrator is not None:
            try:
                await self.orchestrator.stop()
            except Exception:
                pass
        if self.telemetry is not None:
            try:
                await self.telemetry.stop()
            except Exception:
                pass
        try:
            await self.bus.stop()
        except Exception:
            pass
        try:
            await self.audit.stop()
        except Exception:
            pass
        if self.attribution is not None:
            try:
                self.attribution.close()
            except Exception:
                pass
        if self.risk_state is not None:
            try:
                self.risk_state.close()
            except Exception:
                pass
        if self.kill_store is not None:
            try:
                self.kill_store.close()
            except Exception:
                pass
        log.info("daemon.stop")


async def run_daemon(
    *,
    self_check_only: bool = False,
    **kwargs: Any,
) -> int:
    """Start the daemon, run the self-check, and either exit (self-check-only)
    or block on SIGTERM."""
    svc = DaemonService(**kwargs)
    try:
        await svc.start()
        result = await svc.run_startup_self_check()
        if result.get("kind") != "ok":
            log.error("daemon.self_check_failed", result=result)
            return 1
        if self_check_only:
            log.info("daemon.self_check_only_exit", result=result)
            return 0
        await svc.run_forever()
        return 0
    finally:
        await svc.stop()
