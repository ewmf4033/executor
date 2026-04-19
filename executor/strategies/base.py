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
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any

from ..core.events import Event, EventType, Source
from ..core.intent import BasketIntent
from ..core.logging import get_logger


Publish = Callable[[Event], Awaitable[None]]


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
        """
        if self._publish is None:
            raise RuntimeError(
                f"strategy {self.strategy_id} not attached; call executor.register(strategy) first"
            )
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
