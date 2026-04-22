"""
Strategy — abstract base class.

Per Decision 2 of /root/trading-wiki/specs/0d-executor.md:

- strategy_id (unique string)
- required_capabilities: dict[venue -> set of capability flags]
- kelly_cap (Decimal in (0, 1])
- async run() loop (implemented by subclass)
- async emit(intent) method (provided by base; records INTENT_EMITTED,
  enqueues to executor orchestration loop via the event bus)

The executor injects a publish callback at registration time so strategies
do not import the event bus directly.

Phase 4.10 (4.9.1-c): the base also implements a rejection-aware cooldown.
When the same (gate, market_id) pair rejects a strategy's intents above
STRATEGY_REJECT_THRESHOLD times within STRATEGY_REJECT_WINDOW_SEC, the
strategy enters a STRATEGY_REJECT_COOLDOWN_SEC cooldown for that market.
Any INTENT_ADMITTED for that market clears the counters. The daemon
routes GATE_REJECTED / INTENT_ADMITTED events filtered by strategy_id
into `on_gate_rejected` / `on_intent_admitted`.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any

from ..core.events import Event, EventType, Source
from ..core.intent import BasketIntent
from ..core.logging import get_logger


Publish = Callable[[Event], Awaitable[None]]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


# Phase 4.10 (4.9.1-c) defaults.
# Tunable per the 2026-04-22 morning diagnostic (YESNOCrossDetect emitted
# 16,700+ intents / 20h, 100% rejected by market_exposure). Threshold of
# 50/60s ≈ ≤ 1 reject/s sustained before cooldown kicks in; 30s cooldown
# is short enough not to stall real signals but long enough to break
# runaway loops.
STRATEGY_REJECT_THRESHOLD = _env_int("EXECUTOR_STRATEGY_REJECT_THRESHOLD", 50)
STRATEGY_REJECT_WINDOW_SEC = _env_float("EXECUTOR_STRATEGY_REJECT_WINDOW_SEC", 60.0)
STRATEGY_REJECT_COOLDOWN_SEC = _env_float("EXECUTOR_STRATEGY_REJECT_COOLDOWN_SEC", 30.0)


class Strategy(ABC):
    """
    Subclasses must set strategy_id, required_capabilities, kelly_cap and
    implement run().
    """

    strategy_id: str = ""
    required_capabilities: dict[str, frozenset[str]] = {}
    kelly_cap: Decimal = Decimal("0.25")

    def __init__(self) -> None:
        if not self.strategy_id:
            raise ValueError(f"{type(self).__name__}: strategy_id must be set")
        self._publish: Publish | None = None
        self._log = get_logger(f"executor.strategy.{self.strategy_id}")
        # Phase 4.10 (4.9.1-c) rejection cooldown.
        # (gate, market_id) -> deque of reject timestamps (monotonic sec).
        self._reject_history: dict[tuple[str, str], deque[float]] = {}
        # market_id -> monotonic timestamp (sec) when cooldown lifts.
        self._market_cooldown_until: dict[str, float] = {}
        # intent_id -> tuple of market_ids carried by that intent; needed to
        # attribute GATE_REJECTED (which carries intent_id but not market)
        # back to specific markets. Bounded to protect against leaks if
        # ADMIT/REJECT events are lost.
        self._intent_markets: dict[str, tuple[str, ...]] = {}
        self._intent_markets_fifo: deque[str] = deque(maxlen=4096)

    # ------------------------------------------------------------------
    # Market declaration — consumed by RiskPolicy.register_strategy_markets
    # so the StructuralGate knows which (venue, market_id) pairs the
    # strategy can legitimately trade. Phase 4.7: previously only the
    # self-check synthetic markets were registered, causing K1/P1 etc.
    # to be rejected as "market not found".
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def markets(self) -> list[tuple[str, str]]:
        """Return (venue, market_id) pairs this strategy can trade.

        Each subclass must declare its markets. RiskPolicy adds these to
        market_universe so the StructuralGate admits them.
        """

    # ------------------------------------------------------------------
    # Wiring — the executor calls this once at registration.
    # ------------------------------------------------------------------

    def attach(self, publish: Publish) -> None:
        self._publish = publish
        self._log.info("strategy.attach", strategy_id=self.strategy_id)

    # ------------------------------------------------------------------
    # Surface used by subclasses
    # ------------------------------------------------------------------

    async def emit(self, intent: BasketIntent) -> None:
        """
        Publish an INTENT_EMITTED event. Payload includes the full intent
        (legs serialized) and a placeholder for the top-of-book snapshot
        per leg; that snapshot is filled in by the executor orchestration
        loop in Phase 2+. In Phase 1 we record the intent shape only.

        Phase 4.10 (4.9.1-c): if any leg's market is in cooldown from
        repeated same-gate rejections, the emit is dropped with a WARN.
        Strategies that want to avoid the log noise should short-circuit
        earlier by checking `_should_emit_for_market(market_id)`.
        """
        if self._publish is None:
            raise RuntimeError(
                f"strategy {self.strategy_id} not attached; call executor.register(strategy) first"
            )
        # Cooldown check — any leg in cooldown blocks the whole basket.
        now = time.monotonic()
        for leg in intent.legs:
            until = self._market_cooldown_until.get(leg.market_id, 0.0)
            if now < until:
                self._log.warning(
                    "strategy.emit.cooldown_block",
                    strategy_id=self.strategy_id,
                    intent_id=intent.intent_id,
                    market_id=leg.market_id,
                    cooldown_remaining_sec=round(until - now, 3),
                )
                return

        # Record intent -> market_ids for future reject attribution.
        markets = tuple(leg.market_id for leg in intent.legs)
        self._remember_intent(intent.intent_id, markets)

        payload = _serialize_intent(intent)
        event = Event.make(
            EventType.INTENT_EMITTED,
            source=Source.strategy(self.strategy_id),
            payload=payload,
            intent_id=intent.intent_id,
            strategy_id=self.strategy_id,
            # venue/market_id are leg-level for baskets; leave None at intent scope.
        )
        self._log.info(
            "strategy.emit",
            intent_id=intent.intent_id,
            n_legs=len(intent.legs),
            atomicity=intent.atomicity.value,
        )
        await self._publish(event)

    # ------------------------------------------------------------------
    # Phase 4.10 (4.9.1-c) — rejection-aware cooldown.
    # ------------------------------------------------------------------

    def _remember_intent(self, intent_id: str, market_ids: tuple[str, ...]) -> None:
        if intent_id in self._intent_markets:
            return
        self._intent_markets[intent_id] = market_ids
        self._intent_markets_fifo.append(intent_id)
        # If the FIFO reached its cap, an old id was silently evicted —
        # drop its mapping so we don't leak memory.
        if len(self._intent_markets) > self._intent_markets_fifo.maxlen:  # type: ignore[arg-type]
            stale = next(iter(self._intent_markets))
            if stale not in self._intent_markets_fifo:
                self._intent_markets.pop(stale, None)

    def _should_emit_for_market(self, market_id: str, *, now: float | None = None) -> bool:
        """True iff the strategy is not currently in cooldown for `market_id`."""
        now = time.monotonic() if now is None else now
        return now >= self._market_cooldown_until.get(market_id, 0.0)

    def _record_rejection(
        self,
        *,
        gate: str,
        market_id: str,
        now: float | None = None,
    ) -> bool:
        """Returns True iff this rejection just tripped the cooldown."""
        now = time.monotonic() if now is None else now
        key = (gate, market_id)
        history = self._reject_history.setdefault(key, deque())
        # Drop stale entries outside the rolling window.
        cutoff = now - STRATEGY_REJECT_WINDOW_SEC
        while history and history[0] < cutoff:
            history.popleft()
        history.append(now)
        if len(history) > STRATEGY_REJECT_THRESHOLD:
            # Cooldown engages. Reset the counter for this (gate, market)
            # so we don't immediately re-trip when new rejects arrive.
            history.clear()
            self._market_cooldown_until[market_id] = now + STRATEGY_REJECT_COOLDOWN_SEC
            self._log.warning(
                "strategy.cooldown.engage",
                strategy_id=self.strategy_id,
                market_id=market_id,
                gate=gate,
                threshold=STRATEGY_REJECT_THRESHOLD,
                window_sec=STRATEGY_REJECT_WINDOW_SEC,
                cooldown_sec=STRATEGY_REJECT_COOLDOWN_SEC,
            )
            return True
        return False

    def _record_admit(self, market_id: str) -> None:
        """Clear all gate-rejection counters + cooldown for `market_id`."""
        for key in list(self._reject_history.keys()):
            if key[1] == market_id:
                self._reject_history.pop(key, None)
        self._market_cooldown_until.pop(market_id, None)

    async def on_gate_rejected(self, event: Event) -> None:
        """Wired by the daemon: called when GATE_REJECTED matches this strategy."""
        if event.strategy_id != self.strategy_id:
            return
        intent_id = event.intent_id or ""
        markets = self._intent_markets.get(intent_id)
        if not markets:
            return
        payload = event.payload or {}
        gate = str(payload.get("gate", "unknown"))
        for market_id in markets:
            self._record_rejection(gate=gate, market_id=market_id)

    async def on_intent_admitted(self, event: Event) -> None:
        """Wired by the daemon: called when INTENT_ADMITTED matches this strategy."""
        if event.strategy_id != self.strategy_id:
            return
        intent_id = event.intent_id or ""
        markets = self._intent_markets.pop(intent_id, None)
        if not markets:
            return
        for market_id in markets:
            self._record_admit(market_id)

    # ------------------------------------------------------------------
    # Required subclass entry point
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self) -> None:
        """Long-running loop. Subclasses await signals and call self.emit(intent)."""


# ---------------------------------------------------------------------------
# Intent payload serializer — flat, audit-friendly.
# ---------------------------------------------------------------------------


def _serialize_intent(intent: BasketIntent) -> dict[str, Any]:
    return {
        "intent_id": intent.intent_id,
        "strategy_id": intent.strategy_id,
        "atomicity": intent.atomicity.value,
        "max_slippage_per_leg": str(intent.max_slippage_per_leg),
        "basket_target_exposure": str(intent.basket_target_exposure),
        "created_ts": int(intent.created_ts),
        "expires_ts": int(intent.expires_ts),
        "metadata": dict(intent.metadata),
        "legs": [
            {
                "leg_id": leg.leg_id,
                "venue": leg.venue,
                "market_id": leg.market_id,
                "outcome_id": leg.outcome_id,
                "side": leg.side.value,
                "target_exposure": str(leg.target_exposure),
                "price_limit": str(leg.price_limit),
                "confidence": str(leg.confidence),
                "edge_estimate": str(leg.edge_estimate),
                "time_horizon_sec": int(leg.time_horizon_sec),
                "required_capabilities": list(leg.required_capabilities),
                "kelly_fraction_used": str(leg.kelly_fraction_used),
                "metadata": dict(leg.metadata),
                # top-of-book snapshot filled by executor orchestrator in Phase 2+
                "tob_snapshot": None,
            }
            for leg in intent.legs
        ],
    }
