"""
Orchestrator — consumes INTENT_EMITTED events, runs them through the risk
policy, and (on admission) synthesizes paper fills + feeds the attribution
tracker. This is the missing link from Phase 4's burn-in: previously the
strategy emitted intents straight to the audit log with nothing consuming
them.

Paper-fill semantics here are deliberately simple and contained in the
daemon rather than routed through KalshiAdapter:
  - The adapter's paper fill loop exists and is unit-tested; for the
    0d paper daemon we do not depend on a live Kalshi REST to fetch
    orderbooks. Instead, the orchestrator fills each admitted leg at
    its strategy-reported price_limit (the worst price the strategy
    accepts), which is the conservative paper assumption.
  - This keeps the daemon fully offline while still exercising every
    event in the audit pipeline: INTENT_EMITTED → INTENT_ADMITTED (or
    GATE_REJECTED) → FILL → attribution.

When Phase 5 wires a real venue subscription the KalshiAdapter-backed
paper book replaces this synthetic fill path.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from decimal import Decimal
from typing import Any

from ..attribution.tracker import AttributionTracker
from ..audit.writer import AuditWriter
from ..core.event_bus import EventBus
from ..core.events import Event, EventType, Source
from ..core.intent import Atomicity, BasketIntent, Intent, Leg
from ..core.logging import get_logger
from ..core.types import Side
from ..risk.policy import RiskPolicy
from ..risk.types import RiskVerdict


log = get_logger("executor.orchestrator")


def deserialize_intent(payload: dict[str, Any]) -> BasketIntent:
    """Inverse of strategies.base._serialize_intent — strict, single reader."""
    legs = tuple(
        Leg(
            venue=l["venue"],
            market_id=l["market_id"],
            outcome_id=l["outcome_id"],
            side=Side(l["side"]),
            target_exposure=Decimal(str(l["target_exposure"])),
            price_limit=Decimal(str(l["price_limit"])),
            confidence=Decimal(str(l["confidence"])),
            edge_estimate=Decimal(str(l["edge_estimate"])),
            time_horizon_sec=int(l["time_horizon_sec"]),
            required_capabilities=tuple(l.get("required_capabilities") or ()),
            kelly_fraction_used=Decimal(str(l.get("kelly_fraction_used", "0"))),
            leg_id=l["leg_id"],
            metadata=dict(l.get("metadata") or {}),
        )
        for l in payload["legs"]
    )
    return BasketIntent(
        intent_id=payload["intent_id"],
        strategy_id=payload["strategy_id"],
        legs=legs,
        atomicity=Atomicity(payload["atomicity"]),
        max_slippage_per_leg=Decimal(str(payload["max_slippage_per_leg"])),
        basket_target_exposure=Decimal(str(payload["basket_target_exposure"])),
        created_ts=int(payload["created_ts"]),
        expires_ts=int(payload["expires_ts"]),
        metadata=dict(payload.get("metadata") or {}),
    )


class Orchestrator:
    """
    Subscribes to INTENT_EMITTED. For each intent:
      1. deserialize to BasketIntent
      2. policy.evaluate(intent) — which emits GATE_CLIPPED, GATE_REJECTED
         or INTENT_ADMITTED itself
      3. on admit: publish a synthetic FILL event per leg (paper) and feed
         attribution, then publish INTENT_COMPLETE for the basket.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        policy: RiskPolicy,
        attribution: AttributionTracker | None = None,
        audit: AuditWriter | None = None,
        paper_mode: bool = True,
    ) -> None:
        self._bus = bus
        self._policy = policy
        self._attr = attribution
        self._audit = audit
        self._paper_mode = paper_mode
        # Counters — useful for telemetry and the self-check.
        self.n_intents_received = 0
        self.n_admitted = 0
        self.n_rejected = 0
        self.n_filled_legs = 0
        self.n_intent_crashes = 0
        # Phase 4.7 R3: counts crash events that failed BOTH bus and audit.
        self.n_crash_emit_failures = 0
        self._subscription = None

    async def start(self) -> None:
        self._subscription = await self._bus.subscribe(
            "orchestrator",
            on_event=self._on_event,
            queue_maxsize=4096,
        )
        log.info("orchestrator.start", paper_mode=self._paper_mode)

    async def stop(self) -> None:
        await self._bus.unsubscribe("orchestrator")
        log.info("orchestrator.stop")

    async def _on_event(self, event: Event) -> None:
        if event.event_type is not EventType.INTENT_EMITTED:
            return
        try:
            intent = deserialize_intent(event.payload)
        except Exception as exc:
            # Undecodable — never a valid intent, so do NOT increment
            # n_intents_received. Still publish an ERROR so the gap is
            # visible downstream (self-check, audit queries).
            log.error("orchestrator.decode_failed", error=str(exc), event_id=event.event_id)
            payload_intent_id = None
            try:
                payload_intent_id = event.payload.get("intent_id")
            except Exception:
                pass
            await self._emit_crash(
                stage="deserialize",
                intent_id=event.intent_id or payload_intent_id,
                exc=exc,
                strategy_id=event.strategy_id,
            )
            return
        self.n_intents_received += 1
        try:
            if self._attr is not None and intent.legs:
                # Use the YES leg's price_limit as the "decision_price" proxy.
                try:
                    self._attr.note_decision(intent.intent_id, intent.legs[0].price_limit)
                except Exception as exc:
                    self.n_intent_crashes += 1
                    log.error(
                        "orchestrator.note_decision.crash",
                        intent_id=intent.intent_id,
                        error=str(exc),
                    )
                    await self._emit_crash(
                        stage="note_decision",
                        intent_id=intent.intent_id,
                        exc=exc,
                        strategy_id=intent.strategy_id,
                    )
                    return
            verdict: RiskVerdict = await self._policy.evaluate(intent)
        except Exception as exc:
            self.n_intent_crashes += 1
            log.error(
                "orchestrator.risk.crash",
                intent_id=intent.intent_id,
                error=str(exc),
            )
            await self._emit_crash(
                stage="risk_evaluate",
                intent_id=intent.intent_id,
                exc=exc,
                strategy_id=intent.strategy_id,
            )
            return
        if not verdict.admitted:
            self.n_rejected += 1
            # Phase 4.9 Item 3: terminal state — prune attribution caches
            # so rejected intents don't accumulate forever in _decision.
            if self._attr is not None:
                try:
                    self._attr.prune_intent(intent.intent_id)
                except Exception as exc:  # pragma: no cover
                    log.warning(
                        "orchestrator.prune_on_reject_failed",
                        intent_id=intent.intent_id,
                        error=str(exc),
                    )
            return
        self.n_admitted += 1
        if self._paper_mode:
            await self._paper_fill(verdict.intent)
            # Phase 4.9 Item 3: paper fill is immediate/complete → terminal.
            if self._attr is not None:
                try:
                    self._attr.prune_intent(verdict.intent.intent_id)
                except Exception as exc:  # pragma: no cover
                    log.warning(
                        "orchestrator.prune_on_fill_failed",
                        intent_id=verdict.intent.intent_id,
                        error=str(exc),
                    )

    async def _emit_crash(
        self,
        *,
        stage: str,
        intent_id: str | None,
        exc: BaseException,
        strategy_id: str | None = None,
        gate: str | None = None,
    ) -> None:
        # Phase 4.9 Item 3: a crashed intent won't receive further lifecycle
        # events, so prune the attribution caches to avoid leaks.
        if intent_id is not None and self._attr is not None:
            try:
                self._attr.prune_intent(intent_id)
            except Exception:  # pragma: no cover
                pass
        payload: dict[str, Any] = {
            "kind": "ORCHESTRATOR_CRASH",
            "stage": stage,
            "intent_id": intent_id,
            "error": str(exc),
            "exc_type": type(exc).__name__,
        }
        if gate is not None:
            payload["gate"] = gate
        crash_event = Event.make(
            EventType.ERROR,
            source=Source.EXECUTOR,
            intent_id=intent_id,
            strategy_id=strategy_id,
            payload=payload,
        )
        # Phase 4.7 R3: try bus first; if bus.publish raises (stopping,
        # queue full, crash), fall back to synchronous audit write_direct
        # so the crash evidence isn't lost. If both fail, bump counter
        # and log — we've still produced a log line but lost the event.
        try:
            await self._bus.publish(crash_event)
            return
        except Exception as publish_exc:
            log.error(
                "orchestrator.crash_emit.bus_failed",
                error=str(publish_exc),
                intent_id=intent_id,
            )
        if self._audit is not None:
            try:
                self._audit.write_direct(crash_event)
                log.warning(
                    "orchestrator.crash_emit.audit_fallback_ok",
                    intent_id=intent_id,
                )
                return
            except Exception as audit_exc:
                log.error(
                    "orchestrator.crash_emit.audit_failed",
                    error=str(audit_exc),
                    intent_id=intent_id,
                )
        self.n_crash_emit_failures += 1
        log.error(
            "orchestrator.crash_emit.all_paths_failed",
            intent_id=intent_id,
            stage=stage,
            exc_type=type(exc).__name__,
            error=str(exc),
        )

    async def _paper_fill(self, intent: BasketIntent) -> None:
        """
        Synthesize an immediate fill at each leg's price_limit.

        This is the conservative paper assumption: strategies cannot do worse
        than the limit they accepted. It exercises every downstream event
        (FILL → attribution upsert → INTENT_COMPLETE). When a real venue
        subscription lands, the paper poller in KalshiAdapter takes over
        and this method is no longer called.

        Phase 4.11 Item 3 (Review 9 finding #3): re-check the kill switch
        at the start of _paper_fill and before every per-leg emit. The
        policy.kill_switch_recheck gate closes the post-risk gap, but the
        window between INTENT_ADMITTED and first ORDER_PLACED/FILL, as
        well as between legs of a multi-leg basket, is still reachable.
        If killed, we emit ORDER_CANCELLED_PRE_SEND and return without
        placing further orders. Belt-and-suspenders with the policy-side
        recheck — both layers are wanted.
        """
        now_ns = time.time_ns()
        killed, kill_reason = self._policy.kill_switch.is_killed(
            strategy_id=intent.strategy_id, venue=None
        )
        if killed:
            await self._bus.publish(
                Event.make(
                    EventType.ORDER_CANCELLED_PRE_SEND,
                    source=Source.EXECUTOR,
                    intent_id=intent.intent_id,
                    strategy_id=intent.strategy_id,
                    payload={
                        "kill_reason": kill_reason,
                        "intent_id": intent.intent_id,
                        "strategy_id": intent.strategy_id,
                        "n_legs_skipped": len(intent.legs),
                        "stage": "pre_first_leg",
                    },
                )
            )
            log.warning(
                "orchestrator.paper_fill.kill_pre_send",
                intent_id=intent.intent_id,
                reason=kill_reason,
                n_legs_skipped=len(intent.legs),
            )
            return
        filled_legs = 0
        for leg_idx, leg in enumerate(intent.legs):
            # Re-check mid-basket: a kill can fire between legs, and an
            # ALL_OR_NONE basket that's partially emitted produces an
            # orphan. Skip the remaining legs and record how many were
            # skipped so the audit + alerting can reason about it.
            killed, kill_reason = self._policy.kill_switch.is_killed(
                strategy_id=intent.strategy_id, venue=leg.venue
            )
            if killed:
                await self._bus.publish(
                    Event.make(
                        EventType.ORDER_CANCELLED_PRE_SEND,
                        source=Source.EXECUTOR,
                        intent_id=intent.intent_id,
                        strategy_id=intent.strategy_id,
                        payload={
                            "kill_reason": kill_reason,
                            "intent_id": intent.intent_id,
                            "strategy_id": intent.strategy_id,
                            "n_legs_skipped": len(intent.legs) - leg_idx,
                            "n_legs_emitted": leg_idx,
                            "stage": "mid_basket",
                        },
                    )
                )
                log.warning(
                    "orchestrator.paper_fill.kill_mid_basket",
                    intent_id=intent.intent_id,
                    reason=kill_reason,
                    n_legs_skipped=len(intent.legs) - leg_idx,
                    n_legs_emitted=leg_idx,
                )
                break
            fill_id = f"orch-paper-{uuid.uuid4().hex[:16]}"
            order_id = f"orch-order-{uuid.uuid4().hex[:12]}"
            # Emit ORDER_PLACED + FILL so the audit timeline is complete.
            await self._bus.publish(
                Event.make(
                    EventType.ORDER_PLACED,
                    source=Source.EXECUTOR,
                    intent_id=intent.intent_id,
                    leg_id=leg.leg_id,
                    venue=leg.venue,
                    market_id=leg.market_id,
                    strategy_id=intent.strategy_id,
                    payload={
                        "order_id": order_id,
                        "side": leg.side.value,
                        "size": str(leg.target_exposure),
                        "price_limit": str(leg.price_limit),
                        "paper": True,
                    },
                )
            )
            await self._bus.publish(
                Event.make(
                    EventType.FILL,
                    source=Source.venue(leg.venue),
                    intent_id=intent.intent_id,
                    leg_id=leg.leg_id,
                    venue=leg.venue,
                    market_id=leg.market_id,
                    strategy_id=intent.strategy_id,
                    payload={
                        "fill_id": fill_id,
                        "order_id": order_id,
                        "side": leg.side.value,
                        "size": str(leg.target_exposure),
                        "price_prob": str(leg.price_limit),
                        "fee": "0",
                        "ts_ns": now_ns,
                        "paper": True,
                    },
                )
            )
            if self._attr is not None:
                self._attr.note_arrival(intent.intent_id, leg.leg_id, leg.price_limit)
                self._attr.on_fill(
                    fill_id=fill_id,
                    order_id=order_id,
                    intent_id=intent.intent_id,
                    leg_id=leg.leg_id,
                    strategy_id=intent.strategy_id,
                    venue=leg.venue,
                    market_id=leg.market_id,
                    side=leg.side,
                    size=leg.target_exposure,
                    fill_price=leg.price_limit,
                    fill_ts_ns=now_ns,
                    intent_price=leg.price_limit,
                    fee=Decimal("0"),
                    extra={"paper": True, "orchestrator": True},
                )
            filled_legs += 1
        self.n_filled_legs += filled_legs
        await self._bus.publish(
            Event.make(
                EventType.INTENT_COMPLETE,
                source=Source.EXECUTOR,
                intent_id=intent.intent_id,
                strategy_id=intent.strategy_id,
                payload={
                    "n_legs": len(intent.legs),
                    "n_filled": filled_legs,
                    "atomicity": intent.atomicity.value,
                },
            )
        )

    def stats(self) -> dict[str, int]:
        # Invariant: every received intent ends up in exactly one of
        # admitted / rejected / crashed. Violations indicate a dropped
        # intent somewhere in _on_event.
        assert (
            self.n_intents_received
            == self.n_admitted + self.n_rejected + self.n_intent_crashes
        ), (
            f"orchestrator stats invariant broken: "
            f"received={self.n_intents_received} "
            f"admitted={self.n_admitted} "
            f"rejected={self.n_rejected} "
            f"crashed={self.n_intent_crashes}"
        )
        return {
            "intents_received": self.n_intents_received,
            "admitted": self.n_admitted,
            "rejected": self.n_rejected,
            "filled_legs": self.n_filled_legs,
            "intent_crashes": self.n_intent_crashes,
            "crash_emit_failures": self.n_crash_emit_failures,
        }
