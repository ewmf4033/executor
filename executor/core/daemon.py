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
from ..control.socket_server import ControlSocketServer
from ..detectors.adverse_selection import (
    NullAdverseSelectionDetector,
    WindowAdverseSelectionDetector,
)
from ..kill.manager import KillManager
from ..kill.state import KillStateStore
from ..risk.config import ConfigManager, RiskConfig
from ..risk.kill import KillSwitch
from ..risk.policy import RiskPolicy
from ..risk.state import OperatorLivenessStore, RiskState
from ..strategies.yes_no_cross.strategy import CrossPair, YESNOCrossDetect
from ..telegram.bot import TelegramBot
from ..telegram.watchdog import TelegramWatchdog
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
        self.telegram_bot: TelegramBot | None = None
        self.telegram_watchdog: TelegramWatchdog | None = None
        self.control_server: ControlSocketServer | None = None

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
        # Phase 4.10 (4.9.1-a): surface audit-writer fail-closed counters in /pipeline_stats.
        self.telemetry.set_audit_writer(self.audit)
        await self.bus.subscribe("telemetry", on_event=self.telemetry.on_event, queue_maxsize=10_000)
        await self.telemetry.start()

        # Risk subsystem.
        self.config_mgr = ConfigManager(self.risk_yaml if self.risk_yaml.exists() else None)
        # Phase 4.13.1 Fix #B: propagate capital_mode into AuditWriter so
        # audit-write-failure escalation picks HARD + cancel_open_orders
        # under real capital (SOFT-only under paper).
        capital_mode = bool(self.config_mgr.config.capital_mode)
        self.audit.set_capital_mode(capital_mode)
        self.risk_state = RiskState(db_path=self.risk_state_db)
        # Phase 4.13.1 Fix #C: under real capital, refuse to start on a
        # reconstructed cache. Paper mode retains rebuild-from-venues +
        # audit-replay behavior.
        await self.risk_state.load(capital_mode=capital_mode)

        self.kill_store = KillStateStore(self.kill_db)
        # Phase 4.11 Item 1 (Review 8 0c-7, Review 9 #2): share a single
        # KillSwitch instance between KillManager and RiskPolicy so a
        # /kill hard command engages the same object both the gate chain
        # and the post-admission recheck consult. Without this, the
        # manager engaged its own KillSwitch and the policy continued to
        # admit intents via a default, never-engaged instance ("split-brain").
        self._shared_kill_switch = KillSwitch()
        # Phase 4.11 Item 4 (Review 8 0c-6): propagate YAML overrides
        # auto_resume_strike_limit + panic_cooldown_sec into the manager.
        # Previously these were only honored at default values because the
        # config values were never passed through.
        kill_cfg = self.config_mgr.config.kill_switch
        self.kill_mgr = KillManager(
            store=self.kill_store,
            kill_switch=self._shared_kill_switch,
            publish=self.bus.publish,
            auto_resume_strike_limit=kill_cfg.auto_resume_strike_limit,
            panic_cooldown_sec=kill_cfg.panic_cooldown_sec,
        )
        # Phase 4.11 Item 5 (Review 8 0c-5): if the kill DB was corrupt and
        # rebuilt, surface an ERROR event so the operator knows to
        # investigate. The manager exists now so we have a bus reference.
        # Phase 4.13.1 Fix #A: conditionally emit KILL_STATE_FORCE_RESET
        # when the operator used EXECUTOR_FORCE_RESET_KILL_STATE=1 to
        # bypass fail-closed rebuild. Note text varies accordingly so
        # forensics can distinguish force-reset from fail-closed seed.
        if self.kill_store.rebuilt_from_corruption:
            force_reset_used = getattr(self.kill_store, "force_reset_used", False)
            if force_reset_used:
                err_note = (
                    "Corrupt kill_state.sqlite was renamed aside and a fresh "
                    "DB was seeded in mode=NONE via force-reset operator "
                    "override (EXECUTOR_FORCE_RESET_KILL_STATE=1). "
                    "Investigate why the DB became corrupt."
                )
            else:
                err_note = (
                    "Corrupt kill_state.sqlite was renamed aside and "
                    "a fresh DB was seeded in mode=HARD with "
                    "manual_only=True (reason=KILL_DB_CORRUPT_REBUILT). "
                    "Trading is stopped until an operator inspects the "
                    ".corrupt-* backup and explicitly resolves via "
                    "KillManager. Set EXECUTOR_FORCE_RESET_KILL_STATE=1 "
                    "at daemon startup to bypass (seeds NONE instead)."
                )
            await self.bus.publish(
                Event.make(
                    EventType.ERROR,
                    source=Source.KILL,
                    payload={
                        "kind": "KILL_DB_REBUILT_FROM_CORRUPTION",
                        "kill_db_path": str(self.kill_db),
                        "note": err_note,
                    },
                )
            )
            if force_reset_used:
                backup_path = getattr(self.kill_store, "_corruption_backup_path", None)
                await self.bus.publish(
                    Event.make(
                        EventType.KILL_STATE_FORCE_RESET,
                        source=Source.KILL,
                        payload={
                            "kill_db_path": str(self.kill_db),
                            "ns_ts": int(
                                getattr(self.kill_store, "_corruption_ts_ns", 0)
                            ),
                            "backup_path": str(backup_path)
                            if backup_path is not None
                            else "",
                            "note": (
                                "Operator used EXECUTOR_FORCE_RESET_KILL_STATE=1 "
                                "to bypass fail-closed after corruption. "
                                "Investigate why corruption occurred."
                            ),
                        },
                    )
                )
        # Phase 4.9 Item 1: plumb kill_mgr into AuditWriter so persistent
        # audit write failures can engage the kill switch (fail-closed).
        self.audit.set_kill_manager(self.kill_mgr)

        # Phase 4.14a: out-of-process operator control channel. AF_UNIX
        # socket authenticated by filesystem permissions (mode 0600 +
        # User=root). Route kill commands through the already-wired
        # KillManager — same audit trail, same state machine, same
        # fail-closed semantics. No changes to KillManager or the
        # Telegram surface. Socket location defaults to /run/executor
        # (systemd RuntimeDirectory) and is overridable via env var for
        # tests and dev.
        socket_path = os.environ.get(
            "EXECUTOR_CONTROL_SOCKET", "/run/executor/control.sock"
        )
        # Phase 4.14b: operator liveness store shares the risk_state
        # SQLite connection; table is created by RiskState's SCHEMA_SQL
        # and the singleton row is seeded by the store on construction.
        self._operator_liveness = OperatorLivenessStore(
            self.risk_state.connection
        )
        self.control_server = ControlSocketServer(
            socket_path=socket_path,
            kill_mgr=self.kill_mgr,
            daemon_started_ts_ns=time.time_ns(),
            git_sha=os.environ.get("EXECUTOR_GIT_SHA"),
            operator_liveness=self._operator_liveness,
            risk_config_getter=lambda: self.config_mgr.config,
            publish=self.bus.publish,
        )
        await self.control_server.start()

        # Phase 4.11.1: wire TelegramBot into the daemon lifecycle so /kill
        # commands land on the shared KillManager. The bot reads
        # TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID from os.environ at __init__;
        # if either is missing, start() silently no-ops (logs
        # telegram.bot.disabled.no_token / .no_chat_id) and _task stays None.
        self.telegram_bot = TelegramBot(
            kill_manager=self.kill_mgr,
            # Phase 4.14b: same operator_liveness store the control
            # server and risk policy share — Telegram /arm, /heartbeat,
            # etc. must update the same row the DeadManGate reads.
            operator_liveness=self._operator_liveness,
            dead_man_cfg_getter=lambda: self.config_mgr.config.dead_man,
            publish=self.bus.publish,
        )
        await self.telegram_bot.start()
        await self.bus.publish(
            Event.make(
                EventType.EXECUTOR_LIFECYCLE,
                source=Source.EXECUTOR,
                payload={
                    "state": "TELEGRAM_BOT_STARTED",
                    "enabled": self.telegram_bot._task is not None,
                },
            )
        )

        # Phase 4.14c: Telegram polling watchdog. Detects when the
        # getUpdates loop stalls (hung request, dead task) and either
        # restarts the bot or — after repeated failures — engages
        # kill_mgr in SOFT mode. Routed directly through KillManager so
        # the escalation path does not depend on Telegram itself.
        wd_cfg = self.config_mgr.config.telegram.watchdog
        if wd_cfg.enabled:
            self.telegram_watchdog = TelegramWatchdog(
                bot=self.telegram_bot,
                kill_mgr=self.kill_mgr,
                bus=self.bus,
                stall_threshold_sec=wd_cfg.stall_threshold_sec,
                poll_interval_sec=wd_cfg.poll_interval_sec,
                max_restarts=wd_cfg.max_restarts,
                restart_window_sec=wd_cfg.restart_window_sec,
                escalate_on_max=wd_cfg.escalate_on_max,
            )
            self._bg_tasks.append(
                asyncio.create_task(
                    self.telegram_watchdog.run(),
                    name="telegram-watchdog",
                )
            )

        self.attribution = AttributionTracker(
            db_path=self.attribution_db,
            exit_horizon_sec=self.config_mgr.config.attribution.exit_horizon_sec,
            publish=self.bus.publish,
            # Phase 4.11 Item 2 (Review 8 0c-3): settlement records PnL
            # delta against daily_pnl so gate_13 has a real counter.
            risk_state=self.risk_state,
        )

        # Phase 4.7 F4: explicit adverse-selection detector required by
        # RiskPolicy. Use the real WindowAdverseSelectionDetector so the
        # venue-pause plumbing is exercised end-to-end.
        # Phase 4.9 Item 2: pass risk_state so pauses persist to disk.
        self.adverse_selection = WindowAdverseSelectionDetector(
            publish=self.bus.publish,
            state=self.risk_state,
        )
        # Rehydrate any pauses that were active before daemon restart.
        self.adverse_selection.load_from_state(self.risk_state)
        self.policy = RiskPolicy(
            config_manager=self.config_mgr,
            state=self.risk_state,
            adverse_selection=self.adverse_selection,
            publish=self.bus.publish,
            # Phase 4.11 Item 1: share the KillSwitch with KillManager so
            # /kill hard engages the exact instance the gate chain and the
            # kill_switch_recheck both consult.
            kill_switch=self._shared_kill_switch,
            # Phase 4.14b: dead-man gate reads the same operator_liveness
            # store the control server writes to.
            operator_liveness=self._operator_liveness,
        )
        # Phase 4.11 Item 1: invariant assertion — catches regressions
        # where a future refactor forgets to share the instance and silently
        # re-creates a default KillSwitch() in one of the two constructors.
        assert self.kill_mgr._kill_switch is self.policy.kill_switch, (
            "KillSwitch split-brain regression — shared instance invariant broken"
        )
        # Phase 4.11 Item 4: also push kill_switch config updates through
        # SIGHUP reloads. RiskPolicy already registers its own reload hook;
        # ours is additive and runs after, so both subsystems resync.
        async def _kill_mgr_reload(new_cfg: RiskConfig) -> None:
            self.kill_mgr.update_kill_switch_config(
                auto_resume_strike_limit=new_cfg.kill_switch.auto_resume_strike_limit,
                panic_cooldown_sec=new_cfg.kill_switch.panic_cooldown_sec,
            )
        self.config_mgr.register_reload_hook(_kill_mgr_reload)
        self.policy.set_venue_capabilities(
            {"kalshi": frozenset({"supports_limit", "supports_market"}),
             "polymarket": frozenset({"supports_limit"}),
             "self_check_yes": frozenset({"supports_limit"}),
             "self_check_no": frozenset({"supports_limit"})}
        )

        # Phase 4.7 F3: require an orderbook provider OR the
        # EXECUTOR_PAPER_MODE_NO_ORDERBOOK escape hatch. Paper daemon
        # has no real orderbook subscription until Phase 5 wires venue
        # adapters, so it must opt in explicitly.
        allow_no_ob = os.environ.get("EXECUTOR_PAPER_MODE_NO_ORDERBOOK", "").lower() in (
            "1", "true", "yes", "on"
        )
        if self.policy.orderbook_provider is None and not allow_no_ob:
            raise RuntimeError(
                "liquidity provider not configured; "
                "set EXECUTOR_PAPER_MODE_NO_ORDERBOOK=true for paper/test harnesses"
            )

        # Bootstrap the self-check synthetic markets into the policy's
        # market_universe so the StructuralGate admits the synthetic
        # intent deterministically. Phase 4.6: previously the self-check
        # "passed" via GATE_REJECTED short-circuit, masking gate misconfig.
        self.policy.register_self_check_markets()

        # Bus must be pumping before orchestrator subscribes.
        await self.bus.start()

        # Orchestrator: INTENT_EMITTED → risk → paper FILL → attribution.
        # Phase 4.7 R3: pass audit so crash emits can fall back to a
        # direct audit write if the bus is unavailable.
        self.orchestrator = Orchestrator(
            bus=self.bus,
            policy=self.policy,
            attribution=self.attribution,
            audit=self.audit,
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

        # Phase 4.10 (4.9.1-c): route GATE_REJECTED + INTENT_ADMITTED back
        # to the emitting strategy so the cooldown helpers in the base
        # class observe outcomes. Single subscriber; filter by strategy_id
        # on our side rather than adding per-strategy queues.
        async def _strategy_feedback(event: Event) -> None:
            if event.strategy_id != self.strategy.strategy_id:
                return
            if event.event_type == EventType.GATE_REJECTED:
                await self.strategy.on_gate_rejected(event)
            elif event.event_type == EventType.INTENT_ADMITTED:
                await self.strategy.on_intent_admitted(event)
        await self.bus.subscribe(
            "strategy-feedback",
            on_event=_strategy_feedback,
            queue_maxsize=10_000,
        )

        # Phase 4.7 F1 (K1/P1 regression): register the strategy's
        # declared markets so the StructuralGate admits real intents.
        # Additive with register_self_check_markets — both populate the
        # same market_universe set.
        self.policy.register_strategy_markets(self.strategy)

        # Phase 4.7 F5 (event concentration fail-closed): every leg
        # must have an event_id. Map self-check + strategy markets to
        # synthetic event ids here so the gate passes in paper mode.
        # Real event-id wiring (Kalshi series/event lookup) lands in
        # the Phase 5 venue-subscription path.
        event_id_map: dict[tuple[str, str], str] = {
            ("self_check_yes", "SCYES"): "SELF_CHECK_EVENT",
            ("self_check_no", "SCNO"): "SELF_CHECK_EVENT",
        }
        for venue, market_id in self.strategy.markets:
            event_id_map[(venue, market_id)] = f"synth:{venue}:{market_id}"
        self.policy.set_event_id_map(event_id_map)

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
        # Phase 4.9 Item 3: periodic cache sweep for attribution leaks.
        self._bg_tasks.append(
            asyncio.create_task(self._attribution_sweep_loop(), name="attribution-sweep")
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

    async def _attribution_sweep_loop(self) -> None:
        """Phase 4.9 Item 3: periodic orphan pruner. Sweeps any intent whose
        decision/arrival caches are older than EXECUTOR_ATTRIBUTION_MAX_AGE_HOURS
        (default 48h). This is belt-and-suspenders — the orchestrator also
        prunes on terminal states inline; the sweep only catches the edge
        case where the terminal-state emit was missed."""
        assert self.attribution is not None
        try:
            hours_env = os.environ.get("EXECUTOR_ATTRIBUTION_MAX_AGE_HOURS", "48")
            try:
                max_age_sec = float(hours_env) * 3600.0
            except ValueError:
                max_age_sec = 48.0 * 3600.0
            # Run every 5 minutes per spec; first sweep after initial delay
            # so startup isn't noisy.
            interval_sec = 300.0
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=interval_sec
                    )
                    return
                except asyncio.TimeoutError:
                    pass
                try:
                    pruned = self.attribution.prune_older_than(max_age_sec=max_age_sec)
                    if pruned:
                        log.info("daemon.attribution.swept", pruned=pruned)
                except Exception as exc:
                    log.warning("daemon.attribution.sweep_error", error=str(exc))
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Phase 4.7 Q6 shutdown order (drain before unsubscribe):

            1. Signal + cancel strategy/background tasks (no new intents).
            1a. Stop control socket FIRST (Phase 4.14e H1) — no new
                executorctl commands can enter while subsystems tear
                down. Draining the bus or stopping the orchestrator
                while the control socket still accepts arm/kill/resume
                commands would race new state changes against teardown.
            2. Short sleep to let in-flight emits reach the bus.
            3. Publish STOPPING + STATE_SAVED lifecycle events.
            4. Drain the bus inbox so pending events reach orchestrator.
            5. telegram watchdog.stop() (before bot so the watchdog
               does not see bot.stop() as a stall).
            6. telegram bot.stop().
            7. orchestrator.stop() — unsubscribe AFTER drain.
            8. telemetry.stop().
            9. bus.stop() — final close / sentinel delivery.
            10. audit.stop() — flush last (so STATE_SAVED lands).
            11. Close attribution / risk_state / kill_store.
        """
        self._stop_event.set()
        # (1) Cancel background tasks so no new intents are produced.
        for t in self._bg_tasks:
            t.cancel()
        for t in self._bg_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._bg_tasks.clear()

        # (1a) Phase 4.14e — stop the control socket listener immediately
        # after background tasks, BEFORE we drain the bus and tear down
        # subsystems. Any new executorctl connection after this point
        # gets ECONNREFUSED; any in-flight handler is cancelled via
        # server.close() + wait_closed(). Ultrareview Run 2 (H1) flagged
        # the prior ordering — control_server stopped near the end — as
        # allowing kill/resume/arm to land on partially-shutdown state.
        if self.control_server is not None:
            try:
                await self.control_server.stop()
            except Exception:
                pass

        # (2) Brief settle window for any in-flight publishes.
        try:
            await asyncio.sleep(0.05)
        except Exception:
            pass

        # (3) Final lifecycle events before we tear down publishers.
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

        # (4) Drain the bus inbox while orchestrator is still subscribed.
        try:
            drained, timed_out = await self.bus.drain_inbox(timeout_sec=5.0)
            log.info(
                "daemon.stop.drain",
                drained=drained,
                timed_out=timed_out,
            )
        except Exception as exc:
            log.warning("daemon.stop.drain_failed", error=str(exc))

        # (5) Phase 4.14c: stop the Telegram watchdog before the bot so it
        # cannot observe a bot.stop() as a "stall" and attempt a restart
        # during shutdown.
        if self.telegram_watchdog is not None:
            try:
                await self.telegram_watchdog.stop()
            except Exception:
                pass
        # (6) Phase 4.11.1: stop the Telegram bot before closing the bus so
        # its poll loop is torn down cleanly and can't race the sentinel.
        if self.telegram_bot is not None:
            try:
                await self.telegram_bot.stop()
            except Exception:
                pass
        # (7) Now safe to unsubscribe orchestrator.
        if self.orchestrator is not None:
            try:
                await self.orchestrator.stop()
            except Exception:
                pass
        # (8) Telemetry.
        if self.telemetry is not None:
            try:
                await self.telemetry.stop()
            except Exception:
                pass
        # (9) Bus final close.
        try:
            await self.bus.stop()
        except Exception:
            pass
        # (10) Audit flush/close.
        try:
            await self.audit.stop()
        except Exception:
            pass
        # (11) Close backing stores.
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
