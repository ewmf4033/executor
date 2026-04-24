"""
RiskPolicy — orchestrates the 13-gate sequence.

Exit semantics per Decision 3:
- sequential execution in fixed order
- first REJECT wins → emit GATE_REJECTED, return RiskVerdict(admitted=False)
- CLIPs accumulate → emit GATE_CLIPPED after each, replace current_intent
- final survivor → emit INTENT_ADMITTED with gates_passed + gate_timings_ms

Every evaluation ends in exactly one terminal event (ADMITTED or REJECTED).
Clip events precede the terminal event in audit ordering.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable

from ..core.events import Event, EventType, Source
from ..core.intent import BasketIntent, Leg
from ..core.logging import get_logger
from ..detectors.adverse_selection import AdverseSelectionDetector
from ..detectors.poisoning import PoisoningTracker, build_detector
from .config import ConfigManager, RiskConfig
from .exposure import intent_notional_dollars, leg_notional_dollars
from .gates import Gate, default_gate_chain
from .kill import KillSwitch
from .state import OperatorLivenessStore, RiskState
from .types import GateCtx, GateDecision, GateResult, RiskVerdict
from .venue_health import VenueHealth

if TYPE_CHECKING:
    from ..strategies.base import Strategy


log = get_logger("executor.risk.policy")


Publish = Callable[[Event], Awaitable[None]]
OrderbookProvider = Callable[[str, str, str], "Awaitable[Any]"]


class RiskPolicy:
    def __init__(
        self,
        *,
        config_manager: ConfigManager,
        state: RiskState,
        kill_switch: KillSwitch | None = None,
        venue_health: VenueHealth | None = None,
        poisoning: PoisoningTracker | None = None,
        adverse_selection: AdverseSelectionDetector,
        publish: Publish | None = None,
        gates: list[Gate] | None = None,
        operator_liveness: OperatorLivenessStore | None = None,
    ) -> None:
        self._cfg_mgr = config_manager
        self.state = state
        self.kill_switch = kill_switch or KillSwitch()
        cfg = config_manager.config
        self.venue_health = venue_health or VenueHealth(
            window_sec=cfg.venue_health.window_sec,
            trip_threshold=cfg.venue_health.trip_threshold,
            pause_sec=cfg.venue_health.pause_sec,
        )
        if poisoning is None and cfg.poisoning.enabled:
            det = build_detector(
                cfg.poisoning.detector,
                window_sec=cfg.poisoning.window_sec,
                z_threshold=cfg.poisoning.z_threshold,
                min_samples=cfg.poisoning.min_samples,
                **cfg.poisoning.detector_kwargs,
            )
            self.poisoning = PoisoningTracker(
                det, pause_sec=cfg.poisoning.pause_sec, publish=publish
            )
        else:
            self.poisoning = poisoning
        if adverse_selection is None:
            raise ValueError(
                "RiskPolicy: adverse_selection detector is required; "
                "pass NullAdverseSelectionDetector() explicitly for tests/self-check"
            )
        self.adverse_selection = adverse_selection
        self._publish = publish
        # Phase 4.14b: operator liveness store for Gate 8.5 (dead-man).
        # Optional; DeadManGate fails closed when cfg.dead_man.enabled=True
        # but this is None. Tests without a wired store rely on
        # dead_man.enabled defaulting to False.
        self.operator_liveness = operator_liveness
        self.gates: list[Gate] = gates if gates is not None else default_gate_chain()

        # Registries consumed by gates. Populate via setter methods.
        self.market_universe: set[tuple[str, str]] | None = None
        self.venue_capabilities: dict[str, frozenset[str]] = {}
        self._event_id_map: dict[tuple[str, str], str] = {}
        self.orderbook_provider: OrderbookProvider | None = None
        # Phase 4.7 F2: StructuralGate default-denies when market_universe
        # is unset. Tests that don't need a registered universe can flip
        # this via set_allow_universe_bootstrap(True).
        self._allow_universe_bootstrap: bool = False

        # Wire config hooks.
        self._cfg_mgr.register_reload_hook(self._on_config_reload)
        # Seed state's config hash row so tests see it.
        self.state.record_config_hash(self.config.fingerprint())

        self._eval_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Wiring setters
    # ------------------------------------------------------------------

    @property
    def config(self) -> RiskConfig:
        return self._cfg_mgr.config

    def set_publish(self, publish: Publish) -> None:
        self._publish = publish
        if self.poisoning is not None:
            self.poisoning.set_publish(publish)

    def set_market_universe(self, markets: Iterable[tuple[str, str]]) -> None:
        self.market_universe = set(markets)

    def register_self_check_markets(self) -> None:
        """Bootstrap concession for the startup self-check synthetic markets only.

        Ensures (self_check_yes, SCYES) and (self_check_no, SCNO) clear the
        StructuralGate's market-existence check so the synthetic basket is
        deterministically admittable. Called from DaemonService.start()
        before run_startup_self_check. Additive with register_strategy_markets
        — both populate the same market_universe set."""
        if self.market_universe is None:
            self.market_universe = set()
        self.market_universe.add(("self_check_yes", "SCYES"))
        self.market_universe.add(("self_check_no", "SCNO"))

    def register_strategy_markets(self, strategy: "Strategy") -> None:
        """Register a strategy's declared markets into market_universe.

        Additive with register_self_check_markets — populates the same
        set so the StructuralGate admits real strategy markets alongside
        the synthetic self-check ones. Fixes the Phase 4.6 regression
        where K1/P1 intents were rejected as "market not found" because
        only self-check markets had been registered.

        Raises ValueError if the strategy declares no markets — catches
        misconfig at daemon startup rather than silently at first intent.
        """
        markets = strategy.markets
        if not markets:
            raise ValueError(
                f"strategy {strategy.strategy_id} declared no markets; "
                f"at least one (venue, market_id) required"
            )
        if self.market_universe is None:
            self.market_universe = set()
        for venue, market_id in markets:
            self.market_universe.add((venue, market_id))

    def set_allow_universe_bootstrap(self, allow: bool) -> None:
        """Test-only escape hatch: re-enable the pre-Phase-4.7 default-allow
        behavior in StructuralGate when market_universe is None. Logs a
        warning so the bypass is visible. Not called from DaemonService."""
        self._allow_universe_bootstrap = bool(allow)
        if allow:
            log.warning(
                "risk.policy.universe_bootstrap_enabled",
                note="StructuralGate will default-ALLOW when market_universe is None. Test-only.",
            )

    def set_venue_capabilities(self, caps: dict[str, Iterable[str]]) -> None:
        self.venue_capabilities = {v: frozenset(c) for v, c in caps.items()}

    def set_event_id_map(self, mapping: dict[tuple[str, str], str]) -> None:
        self._event_id_map = dict(mapping)

    def set_orderbook_provider(self, provider: OrderbookProvider) -> None:
        self.orderbook_provider = provider

    def event_id_for(self, venue: str, market_id: str) -> str | None:
        return self._event_id_map.get((venue, market_id))

    # ------------------------------------------------------------------
    # Config reload hook
    # ------------------------------------------------------------------

    async def _on_config_reload(self, new_cfg: RiskConfig) -> None:
        self.venue_health.update_from_config(
            window_sec=new_cfg.venue_health.window_sec,
            trip_threshold=new_cfg.venue_health.trip_threshold,
            pause_sec=new_cfg.venue_health.pause_sec,
        )
        if self.poisoning is not None:
            self.poisoning.update_pause_sec(new_cfg.poisoning.pause_sec)
        self.state.record_config_hash(new_cfg.fingerprint())
        if self._publish is not None:
            await self._publish(
                Event.make(
                    EventType.CONFIG_RELOADED,
                    source=Source.RISK,
                    payload={
                        "hash": new_cfg.fingerprint(),
                        "path": str(self._cfg_mgr.path) if self._cfg_mgr.path else None,
                        "loaded_ts_ns": self._cfg_mgr.last_reload_ts_ns,
                    },
                )
            )

    # ------------------------------------------------------------------
    # Evaluate — the hot path
    # ------------------------------------------------------------------

    async def evaluate(self, intent: BasketIntent) -> RiskVerdict:
        async with self._eval_lock:
            ctx = GateCtx(
                original_intent=intent,
                current_intent=intent,
                policy=self,
                now_ns=time.time_ns(),
            )
            for gate in self.gates:
                t0 = time.perf_counter()
                try:
                    result = await gate.check(ctx)
                except Exception as exc:
                    log.error("risk.gate.crash", gate=gate.name, error=str(exc))
                    result = GateResult.reject(f"gate {gate.name} crashed: {exc}")
                t_ms = (time.perf_counter() - t0) * 1000.0
                ctx.gate_timings_ms[gate.name] = t_ms

                if result.decision == GateDecision.APPROVE:
                    ctx.gates_passed.append(gate.name)
                    continue

                if result.decision == GateDecision.CLIP:
                    await self._apply_clip(gate, result, ctx)
                    ctx.gates_passed.append(gate.name)
                    continue

                # REJECT
                await self._emit_reject(gate, result, ctx)
                return RiskVerdict(
                    admitted=False,
                    intent=ctx.current_intent,
                    gates_passed=tuple(ctx.gates_passed),
                    gate_timings_ms=dict(ctx.gate_timings_ms),
                    clip_history=tuple(ctx.clip_history),
                    reject_reason=result.reason,
                    reject_gate=gate.name,
                )

            # Phase 4.7 F6: re-check kill switch after the gate loop.
            # KillGate at gate 2 fails fast, but the remaining 12 gates can
            # take 10-100ms — a kill signal that fires during that window
            # must not allow admission.
            for leg in ctx.current_intent.legs:
                killed, reason = self.kill_switch.is_killed(
                    strategy_id=ctx.current_intent.strategy_id, venue=leg.venue
                )
                if killed:
                    await self._emit_reject(
                        _KillSwitchRecheckGate(),
                        GateResult.reject(f"kill_switch: {reason}"),
                        ctx,
                    )
                    return RiskVerdict(
                        admitted=False,
                        intent=ctx.current_intent,
                        gates_passed=tuple(ctx.gates_passed),
                        gate_timings_ms=dict(ctx.gate_timings_ms),
                        clip_history=tuple(ctx.clip_history),
                        reject_reason=f"kill_switch: {reason}",
                        reject_gate="kill_switch_recheck",
                    )

            # All gates passed.
            await self._emit_admitted(ctx)
            self._apply_exposure_side_effects(ctx.current_intent)
            return RiskVerdict(
                admitted=True,
                intent=ctx.current_intent,
                gates_passed=tuple(ctx.gates_passed),
                gate_timings_ms=dict(ctx.gate_timings_ms),
                clip_history=tuple(ctx.clip_history),
                reject_reason=None,
                reject_gate=None,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _apply_clip(self, gate: Gate, result: GateResult, ctx: GateCtx) -> None:
        new_sizes = result.new_leg_sizes or {}
        intent = ctx.current_intent
        new_legs: list[Leg] = []
        clip_record_per_leg: list[dict[str, Any]] = []
        for leg in intent.legs:
            new_size = new_sizes.get(leg.leg_id, leg.target_exposure)
            if new_size != leg.target_exposure:
                clip_record_per_leg.append(
                    {
                        "gate": gate.name,
                        "leg_id": leg.leg_id,
                        "venue": leg.venue,
                        "market_id": leg.market_id,
                        "original": str(leg.target_exposure),
                        "clipped": str(new_size),
                    }
                )
            new_legs.append(replace(leg, target_exposure=new_size))
        clipped_intent = replace(intent, legs=tuple(new_legs))
        ctx.current_intent = clipped_intent
        ctx.clip_history.extend(clip_record_per_leg)
        # Persist + emit.
        for rec in clip_record_per_leg:
            self.state.record_clip(
                intent_id=intent.intent_id,
                gate=gate.name,
                original=Decimal(rec["original"]),
                clipped=Decimal(rec["clipped"]),
                now_ns=ctx.now_ns,
            )
        if self._publish is not None:
            await self._publish(
                Event.make(
                    EventType.GATE_CLIPPED,
                    source=Source.RISK,
                    intent_id=intent.intent_id,
                    strategy_id=intent.strategy_id,
                    payload={
                        "gate": gate.name,
                        "gate_order": gate.order,
                        "reason": result.reason,
                        "clips": clip_record_per_leg,
                        "gate_ms": ctx.gate_timings_ms[gate.name],
                    },
                )
            )

    async def _emit_reject(
        self, gate: Gate, result: GateResult, ctx: GateCtx
    ) -> None:
        if self._publish is None:
            return
        await self._publish(
            Event.make(
                EventType.GATE_REJECTED,
                source=Source.RISK,
                intent_id=ctx.original_intent.intent_id,
                strategy_id=ctx.original_intent.strategy_id,
                payload={
                    "gate": gate.name,
                    "gate_order": gate.order,
                    "reason": result.reason,
                    "gates_passed": list(ctx.gates_passed),
                    "gate_timings_ms": dict(ctx.gate_timings_ms),
                    "gate_ms": ctx.gate_timings_ms.get(gate.name, 0.0),
                },
            )
        )

    async def _emit_admitted(self, ctx: GateCtx) -> None:
        if self._publish is None:
            return
        final_size = intent_notional_dollars(ctx.current_intent)
        original_size = intent_notional_dollars(ctx.original_intent)
        await self._publish(
            Event.make(
                EventType.INTENT_ADMITTED,
                source=Source.RISK,
                intent_id=ctx.original_intent.intent_id,
                strategy_id=ctx.original_intent.strategy_id,
                payload={
                    "gates_passed": list(ctx.gates_passed),
                    "gate_timings_ms": dict(ctx.gate_timings_ms),
                    "final_dollars": str(final_size),
                    "original_dollars": str(original_size),
                    "clip_history": list(ctx.clip_history),
                    "legs": [
                        {
                            "leg_id": leg.leg_id,
                            "venue": leg.venue,
                            "market_id": leg.market_id,
                            "outcome_id": leg.outcome_id,
                            "side": leg.side.value,
                            "size": str(leg.target_exposure),
                            "price_limit": str(leg.price_limit),
                        }
                        for leg in ctx.current_intent.legs
                    ],
                },
            )
        )

    def _apply_exposure_side_effects(self, intent: BasketIntent) -> None:
        """Book committed exposure so subsequent gate evaluations see it."""
        for leg in intent.legs:
            if leg.target_exposure <= 0:
                continue
            dollars = leg_notional_dollars(leg)
            event_id = self.event_id_for(leg.venue, leg.market_id)
            self.state.add_exposure(
                venue=leg.venue,
                market_id=leg.market_id,
                outcome_id=leg.outcome_id,
                dollars=dollars,
                event_id=event_id,
            )
        self.state.add_strategy_exposure(
            intent.strategy_id, intent_notional_dollars(intent)
        )


class _KillSwitchRecheckGate:
    """Marker-gate used to emit GATE_REJECTED with a recognizable name
    when the kill switch fires between the gate loop and admission.
    Not part of the normal chain — instantiated only at reject time so
    the audit payload carries gate=kill_switch_recheck."""

    name = "kill_switch_recheck"
    order = "2.post"
