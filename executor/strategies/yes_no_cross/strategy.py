"""
YESNOCrossDetect — proof-of-life cross-venue arb detector.

Listens to quote updates (best ask of YES on one venue, best ask of NO on
another). When YES.ask + NO.ask < (1 - fee_buffer), emits an ALL_OR_NONE
BasketIntent: BUY YES on venue_a + BUY NO on venue_b. If both legs fill,
the basket is locked into a 1-contract risk-free payoff (minus fees).

Sizing: 1/4 Kelly at the basket level. With edge = (1 - fee_buffer) - sum
and a "conservative bankroll" of 1.0 unit, basket size in contracts is:

    f_kelly = edge / 1.0          (treat each contract as a unit binary)
    f_used  = f_kelly * 0.25
    contracts = floor(bankroll * f_used / max_leg_cost)

For Phase 4 we just emit a small fixed contract count derived from
(edge * kelly_cap) so risk gates have something concrete to evaluate.
The strategy is paper-mode only; no real orders are placed.

Quote intake:
- accept_quote(venue, market_id, outcome_id, best_ask, mid)
- attempt_emit() — checks current snapshot for cross-venue mispricing and
  emits a basket intent if found.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ...core.intent import Atomicity, BasketIntent, Intent, Leg
from ...core.types import Side
from ..base import Strategy


DEFAULT_FEE_BUFFER = Decimal("0.02")    # treat 2c as round-trip fees
DEFAULT_THRESHOLD = Decimal("0.98")     # YES.ask + NO.ask must be < 0.98
MIN_CONTRACTS = Decimal("1")
MAX_CONTRACTS = Decimal("100")


@dataclass
class _BookCorner:
    venue: str
    market_id: str
    outcome_id: str             # "YES" or "NO"
    best_ask: Decimal
    mid: Decimal
    ts_ns: int


@dataclass
class CrossPair:
    """Pair definition: which venue carries YES and which carries NO."""
    yes_venue: str
    yes_market_id: str
    no_venue: str
    no_market_id: str

    def key(self) -> tuple[str, str, str, str]:
        return (self.yes_venue, self.yes_market_id, self.no_venue, self.no_market_id)


class YESNOCrossDetect(Strategy):
    strategy_id = "yes_no_cross"
    required_capabilities = {
        # Each venue needs limit orders. Set per-venue when registering.
    }
    kelly_cap = Decimal("0.25")

    def __init__(
        self,
        *,
        pairs: list[CrossPair],
        fee_buffer: Decimal = DEFAULT_FEE_BUFFER,
        threshold: Decimal = DEFAULT_THRESHOLD,
        max_contracts: Decimal = MAX_CONTRACTS,
        emit_cooldown_sec: float = 5.0,
    ) -> None:
        super().__init__()
        self._pairs = list(pairs)
        self._fee_buffer = Decimal(str(fee_buffer))
        self._threshold = Decimal(str(threshold))
        self._max_contracts = Decimal(str(max_contracts))
        self._emit_cooldown_sec = emit_cooldown_sec
        self._book: dict[tuple[str, str, str], _BookCorner] = {}
        self._last_emit_ts: dict[tuple[str, str, str, str], float] = {}
        self._stop = asyncio.Event()
        # Allow tests to inject a quote feed via accept_quote without running
        # the full poll loop.

    # ------------------------------------------------------------------
    # Quote intake
    # ------------------------------------------------------------------

    @property
    def markets(self) -> list[tuple[str, str]]:
        """Both legs of each cross pair: YES venue and NO venue."""
        out: list[tuple[str, str]] = []
        for pair in self._pairs:
            out.append((pair.yes_venue, pair.yes_market_id))
            out.append((pair.no_venue, pair.no_market_id))
        return out

    def accept_quote(
        self,
        *,
        venue: str,
        market_id: str,
        outcome_id: str,
        best_ask: Decimal,
        mid: Decimal,
        ts_ns: int | None = None,
    ) -> None:
        self._book[(venue, market_id, outcome_id)] = _BookCorner(
            venue=venue,
            market_id=market_id,
            outcome_id=outcome_id,
            best_ask=Decimal(str(best_ask)),
            mid=Decimal(str(mid)),
            ts_ns=ts_ns or time.time_ns(),
        )

    # ------------------------------------------------------------------
    # Emit-side
    # ------------------------------------------------------------------

    def find_cross(self, pair: CrossPair) -> tuple[Decimal, Decimal, Decimal] | None:
        """Returns (sum, edge, size) if a basket should fire, else None."""
        yes = self._book.get((pair.yes_venue, pair.yes_market_id, "YES"))
        no_ = self._book.get((pair.no_venue, pair.no_market_id, "NO"))
        if yes is None or no_ is None:
            return None
        sum_ask = yes.best_ask + no_.best_ask
        if sum_ask >= self._threshold:
            return None
        # Edge in probability units. Use threshold (post-fee budget).
        edge = self._threshold - sum_ask
        # Basket-level Kelly sizing.
        f_used = edge * self.kelly_cap
        # Conservative bankroll = 1 unit per contract; size cap by max_contracts.
        size = (Decimal("100") * f_used).quantize(Decimal("1"))
        if size < MIN_CONTRACTS:
            size = MIN_CONTRACTS
        if size > self._max_contracts:
            size = self._max_contracts
        return sum_ask, edge, size

    def build_intent(self, pair: CrossPair) -> BasketIntent | None:
        cross = self.find_cross(pair)
        if cross is None:
            return None
        sum_ask, edge, size = cross
        yes = self._book[(pair.yes_venue, pair.yes_market_id, "YES")]
        no_ = self._book[(pair.no_venue, pair.no_market_id, "NO")]
        now = time.time_ns()
        legs = (
            Leg(
                venue=pair.yes_venue,
                market_id=pair.yes_market_id,
                outcome_id="YES",
                side=Side.BUY,
                target_exposure=size,
                price_limit=yes.best_ask,
                confidence=Decimal("0.85"),
                edge_estimate=edge,
                time_horizon_sec=600,
                required_capabilities=("supports_limit",),
                kelly_fraction_used=self.kelly_cap,
                metadata={"role": "YES", "best_ask": str(yes.best_ask)},
            ),
            Leg(
                venue=pair.no_venue,
                market_id=pair.no_market_id,
                outcome_id="NO",
                side=Side.BUY,
                target_exposure=size,
                price_limit=no_.best_ask,
                confidence=Decimal("0.85"),
                edge_estimate=edge,
                time_horizon_sec=600,
                required_capabilities=("supports_limit",),
                kelly_fraction_used=self.kelly_cap,
                metadata={"role": "NO", "best_ask": str(no_.best_ask)},
            ),
        )
        # Basket-level $ exposure ~ size * 1.0 (max payoff per pair is $1).
        return Intent.basket(
            strategy_id=self.strategy_id,
            legs=legs,
            atomicity=Atomicity.ALL_OR_NONE,
            max_slippage_per_leg=Decimal("0.02"),
            basket_target_exposure=size,
            created_ts=now,
            expires_ts=now + 30_000_000_000,   # 30s expiry
            metadata={
                "pair": list(pair.key()),
                "sum_ask": str(sum_ask),
                "edge": str(edge),
                "fee_buffer": str(self._fee_buffer),
                "threshold": str(self._threshold),
            },
        )

    async def attempt_emit(self) -> list[BasketIntent]:
        """Walk all pairs and emit baskets where edge exists. Honors per-pair cooldown."""
        emitted: list[BasketIntent] = []
        now = time.monotonic()
        for pair in self._pairs:
            last = self._last_emit_ts.get(pair.key(), 0.0)
            if now - last < self._emit_cooldown_sec:
                continue
            intent = self.build_intent(pair)
            if intent is None:
                continue
            await self.emit(intent)
            self._last_emit_ts[pair.key()] = now
            emitted.append(intent)
        return emitted

    # ------------------------------------------------------------------
    # run() — used when the strategy is hosted by a real executor loop.
    # The integration test bypasses this and calls accept_quote / attempt_emit
    # directly so it doesn't need real WebSocket plumbing.
    # ------------------------------------------------------------------

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.attempt_emit()
            except Exception as exc:  # pragma: no cover
                self._log.warning("yes_no_cross.emit.error", error=str(exc))
            await asyncio.sleep(1.0)

    async def stop(self) -> None:
        self._stop.set()
