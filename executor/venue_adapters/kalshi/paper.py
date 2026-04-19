"""
Paper-mode order book for the Kalshi adapter.

When PAPER_MODE is on, place_limit / place_market / replace_order /
cancel_order do NOT call Kalshi's trade endpoints. They mutate this
in-memory store and emit synthetic FillEvents when a paper limit order
would cross the live orderbook fetched from get_orderbook.

Fill semantics (intentionally simple, expandable in Phase 3+):
  - BUY  fills when paper limit price >= best ask of (canonical) outcome.
  - SELL fills when paper limit price <= best bid of (canonical) outcome.
  - Market orders fill immediately at best ask (BUY) / best bid (SELL),
    walking up to the requested size against the level sizes.
  - Fill price is the *resting* level price, not the paper limit price
    (price improvement when the book is deeper than the limit).
  - Fees are zero in paper. Real Kalshi fee math lives in the live path.
  - One fill per check; partial fills allowed if level size < order size,
    remainder stays open and is tried again on the next orderbook tick.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ...core.types import Fill, OrderState, OrderStatus, Side, TIF


def _now_ns() -> int:
    return time.time_ns()


@dataclass
class _PaperOrder:
    order_id: str
    market_id: str
    outcome_id: str       # canonical "YES" / "NO"
    side: Side
    size: Decimal
    filled: Decimal
    price_prob: Decimal | None     # None = market order
    tif: TIF | None
    placed_ts_ns: int
    last_update_ts_ns: int
    state: OrderState = OrderState.OPEN
    fills: list[Fill] = field(default_factory=list)

    def remaining(self) -> Decimal:
        return self.size - self.filled


class PaperBook:
    """In-memory paper order/fill store. One per KalshiAdapter instance."""

    def __init__(self) -> None:
        self._orders: dict[str, _PaperOrder] = {}

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def place_limit(
        self,
        market_id: str,
        outcome_id: str,
        side: Side,
        size: Decimal,
        price_prob: Decimal,
        tif: TIF,
    ) -> str:
        order_id = f"paper-{uuid.uuid4().hex[:16]}"
        now = _now_ns()
        self._orders[order_id] = _PaperOrder(
            order_id=order_id,
            market_id=market_id,
            outcome_id=outcome_id,
            side=side,
            size=Decimal(str(size)),
            filled=Decimal("0"),
            price_prob=Decimal(str(price_prob)),
            tif=tif,
            placed_ts_ns=now,
            last_update_ts_ns=now,
        )
        return order_id

    def place_market(
        self,
        market_id: str,
        outcome_id: str,
        side: Side,
        size: Decimal,
    ) -> str:
        order_id = f"paper-{uuid.uuid4().hex[:16]}"
        now = _now_ns()
        self._orders[order_id] = _PaperOrder(
            order_id=order_id,
            market_id=market_id,
            outcome_id=outcome_id,
            side=side,
            size=Decimal(str(size)),
            filled=Decimal("0"),
            price_prob=None,
            tif=TIF.IOC,
            placed_ts_ns=now,
            last_update_ts_ns=now,
        )
        return order_id

    def cancel(self, order_id: str) -> bool:
        o = self._orders.get(order_id)
        if o is None:
            return False
        if o.state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED):
            return False
        o.state = OrderState.CANCELLED
        o.last_update_ts_ns = _now_ns()
        return True

    def replace(
        self,
        order_id: str,
        new_price: Decimal | None,
        new_size: Decimal | None,
    ) -> str:
        o = self._orders.get(order_id)
        if o is None:
            raise KeyError(order_id)
        if o.state not in (OrderState.OPEN, OrderState.PARTIAL, OrderState.PENDING):
            raise ValueError(f"order {order_id} not replaceable in state {o.state}")
        # Replacement convention: cancel old, mint new id with merged params,
        # carry forward filled qty.
        new_order_id = f"paper-{uuid.uuid4().hex[:16]}"
        now = _now_ns()
        new_order = _PaperOrder(
            order_id=new_order_id,
            market_id=o.market_id,
            outcome_id=o.outcome_id,
            side=o.side,
            size=Decimal(str(new_size)) if new_size is not None else o.size,
            filled=o.filled,
            price_prob=Decimal(str(new_price)) if new_price is not None else o.price_prob,
            tif=o.tif,
            placed_ts_ns=o.placed_ts_ns,
            last_update_ts_ns=now,
            state=OrderState.OPEN if o.filled < (Decimal(str(new_size)) if new_size else o.size) else OrderState.FILLED,
        )
        o.state = OrderState.CANCELLED
        o.last_update_ts_ns = now
        self._orders[new_order_id] = new_order
        return new_order_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, order_id: str) -> _PaperOrder | None:
        return self._orders.get(order_id)

    def status(self, order_id: str) -> OrderStatus:
        o = self._orders.get(order_id)
        if o is None:
            raise KeyError(order_id)
        return OrderStatus(
            order_id=o.order_id,
            venue="kalshi",
            market_id=o.market_id,
            outcome_id=o.outcome_id,
            side=o.side,
            state=o.state,
            size=o.size,
            filled=o.filled,
            price_prob=o.price_prob,
            tif=o.tif,
            placed_ts_ns=o.placed_ts_ns,
            last_update_ts_ns=o.last_update_ts_ns,
            native={"paper": True},
        )

    def fills_since(self, since_ts_ns: int) -> list[Fill]:
        out: list[Fill] = []
        for o in self._orders.values():
            for f in o.fills:
                if f.ts_ns >= since_ts_ns:
                    out.append(f)
        out.sort(key=lambda f: f.ts_ns)
        return out

    def open_orders(self, market_id: str | None = None) -> list[_PaperOrder]:
        out = []
        for o in self._orders.values():
            if o.state not in (OrderState.OPEN, OrderState.PARTIAL, OrderState.PENDING):
                continue
            if market_id is not None and o.market_id != market_id:
                continue
            out.append(o)
        return out

    # ------------------------------------------------------------------
    # Fill simulation against a live orderbook
    # ------------------------------------------------------------------

    def try_fill_against(self, orderbook: Any) -> list[Fill]:
        """
        Walk every open order on the orderbook's market+outcome and emit fills
        for any whose price now crosses. Returns the new Fills (also recorded
        on the order). Caller is responsible for publishing the FillEvents.
        """
        new_fills: list[Fill] = []
        market_id = orderbook.market_id
        outcome_id = orderbook.outcome_id
        for o in list(self._orders.values()):
            if o.market_id != market_id or o.outcome_id != outcome_id:
                continue
            if o.state not in (OrderState.OPEN, OrderState.PARTIAL):
                continue
            new_fills.extend(self._fill_one(o, orderbook))
        return new_fills

    def _fill_one(self, order: _PaperOrder, orderbook: Any) -> list[Fill]:
        """Fill order against the appropriate side of orderbook. Mutates order."""
        out: list[Fill] = []
        # BUY consumes asks (lowest ask first); SELL consumes bids (highest bid first).
        levels = orderbook.asks if order.side == Side.BUY else orderbook.bids
        if not levels:
            return out

        for level in levels:
            remaining = order.remaining()
            if remaining <= 0:
                break
            # Price gate.
            if order.price_prob is not None:
                if order.side == Side.BUY and level.price_prob > order.price_prob:
                    break  # asks are ascending, no further crosses
                if order.side == Side.SELL and level.price_prob < order.price_prob:
                    break  # bids are descending, no further crosses
            take = remaining if level.size >= remaining else level.size
            if take <= 0:
                continue
            fill = Fill(
                fill_id=f"paper-fill-{uuid.uuid4().hex[:16]}",
                order_id=order.order_id,
                venue="kalshi",
                market_id=order.market_id,
                outcome_id=order.outcome_id,
                side=order.side,
                size=Decimal(str(take)),
                price_prob=level.price_prob,
                fee=Decimal("0"),
                ts_ns=_now_ns(),
                native_quote={"cents": int(level.price_prob * 100)},
                native={"paper": True},
            )
            order.fills.append(fill)
            order.filled += take
            order.last_update_ts_ns = fill.ts_ns
            out.append(fill)

        if order.filled >= order.size:
            order.state = OrderState.FILLED
        elif order.filled > 0:
            order.state = OrderState.PARTIAL
        return out
