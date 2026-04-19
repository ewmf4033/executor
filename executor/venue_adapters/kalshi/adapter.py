"""
KalshiAdapter — concrete VenueAdapter for api.elections.kalshi.com.

17 operations (Decision 1 + post-Phase 1 get_positions clarification):

Market data:        get_markets, get_market_spec, get_orderbook
Account:            get_account, get_capabilities, get_positions
Order lifecycle:    place_limit, place_market, place_stop (NotSupported),
                    replace_order, cancel_order, get_order_status, get_fills
Subscriptions:      subscribe_orderbook, subscribe_trades,
                    subscribe_fills, subscribe_positions

PAPER_MODE (env var, default 'true'):
  - place_limit/place_market/replace_order/cancel_order DO NOT call Kalshi
    trade endpoints. They mutate an in-process PaperBook and emit synthetic
    FILL events via a paper-fill loop that polls live orderbooks.
  - All read-only operations (markets, orderbook, balance, positions, orders,
    fills, all subscribe_*) ALWAYS run live against Kalshi.

Canonical price representation per Decision 1: Decimal in [0, 1], outcome_id
"YES"/"NO" for binary markets, native cents preserved in order/fill metadata.
"""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Iterable
from decimal import Decimal
from typing import Any

from ...core.logging import get_logger
from ...core.types import (
    Account,
    Capabilities,
    Fill,
    FillEvent,
    InvalidPrice,
    Market,
    MarketSpec,
    NotSupportedError,
    Orderbook,
    OrderbookEvent,
    OrderState,
    OrderStatus,
    Position,
    PositionEvent,
    Side,
    TIF,
    TradeEvent,
)
from ..base import VenueAdapter
from .auth import KalshiAuth, auth_from_env
from .convert import (
    cents_to_prob,
    canonicalize_outcome,
    parse_orderbook,
    prob_to_cents,
    side_to_action,
    to_native_outcome,
)
from .paper import PaperBook
from .rest import KalshiREST, DEFAULT_BASE_URL
from .websocket import KalshiWS, DEFAULT_WS_URL, DEFAULT_WS_PATH


log = get_logger("executor.venue.kalshi")


def _env_paper_mode() -> bool:
    raw = os.environ.get("PAPER_MODE", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


class KalshiAdapter(VenueAdapter):
    venue_id = "kalshi"

    def __init__(
        self,
        *,
        auth: KalshiAuth | None = None,
        base_url: str | None = None,
        ws_url: str | None = None,
        paper_mode: bool | None = None,
        rest: KalshiREST | None = None,
        ws: KalshiWS | None = None,
        paper_fill_poll_sec: float = 1.0,
    ) -> None:
        self._paper_mode = _env_paper_mode() if paper_mode is None else paper_mode
        self._auth = auth or (auth_from_env() if not rest else None)
        self._base_url = base_url or os.environ.get("KALSHI_BASE_URL", DEFAULT_BASE_URL)
        self._ws_url = ws_url or os.environ.get("KALSHI_WS_URL", DEFAULT_WS_URL)
        self._rest = rest or KalshiREST(self._auth, base_url=self._base_url)
        self._ws = ws  # lazy-built; subscribers need real auth even in paper mode for live data
        self._paper = PaperBook()
        self._paper_fill_poll_sec = paper_fill_poll_sec
        # Per-market paper poller tasks + fill broadcast queue.
        self._paper_pollers: dict[str, asyncio.Task[None]] = {}
        self._paper_fill_queues: list[asyncio.Queue[Fill]] = []
        self._closed = False
        log.info("kalshi.adapter.init", paper_mode=self._paper_mode, base_url=self._base_url)

    @property
    def paper_mode(self) -> bool:
        return self._paper_mode

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for t in self._paper_pollers.values():
            t.cancel()
        for t in self._paper_pollers.values():
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._paper_pollers.clear()
        if self._ws is not None:
            await self._ws.stop()
        await self._rest.close()

    # ==================================================================
    # Market data
    # ==================================================================

    async def get_markets(self) -> list[Market]:
        """List currently-open markets. Paginates through cursor."""
        out: list[Market] = []
        cursor: str | None = None
        pages = 0
        while True:
            params: dict[str, Any] = {"status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = await self._rest.get_markets(**params)
            for raw in data.get("markets", []) or []:
                out.append(_market_from_raw(raw))
            cursor = data.get("cursor")
            pages += 1
            if not cursor or pages > 50:
                break
        return out

    async def get_market_spec(self, market_id: str) -> MarketSpec:
        data = await self._rest.get_market(market_id)
        raw = data.get("market") or data
        # Kalshi binary tick is 1 cent = 0.01 prob. lot_size is 1 contract.
        # min_notional varies by market; default to 1 contract * 1 cent = $0.01.
        # Fees: Kalshi uses a per-share formula; we record the canonical bps
        # estimate as 0 here and stash the real native blob.
        return MarketSpec(
            market_id=market_id,
            venue="kalshi",
            tick_size=Decimal("0.01"),
            lot_size=Decimal("1"),
            min_notional=Decimal("0.01"),
            fees_bps=Decimal("0"),
            yes_no_mapping={"YES": "yes", "NO": "no"},
            native=raw,
        )

    async def get_orderbook(self, market_id: str) -> Orderbook:
        data = await self._rest.get_orderbook(market_id)
        return parse_orderbook(market_id, data, outcome="YES")

    # ==================================================================
    # Account / capabilities / positions
    # ==================================================================

    async def get_account(self) -> Account:
        data = await self._rest.get_balance()
        # Kalshi returns balance in cents. We flatten to dollars (Decimal).
        balance_cents = int(data.get("balance", 0))
        cash = (Decimal(balance_cents) / Decimal(100)).quantize(Decimal("0.01"))
        return Account(
            cash=cash,
            unrealized_pnl=Decimal("0"),
            total_exposure=Decimal("0"),
            free_capital=cash,
            currency="USD",
            as_of_ts=time.time_ns(),
            native=data,
        )

    async def get_capabilities(self) -> Capabilities:
        return Capabilities(
            venue="kalshi",
            supports_limit=True,
            supports_market=True,
            supports_stop=False,
            supports_replace=True,
            supports_orderbook_stream=True,
            supports_trade_stream=True,
            supports_fill_stream=True,
            supports_position_stream=True,
            min_tick=Decimal("0.01"),
            atomicity_scope="SINGLE",
            extra={"paper_mode": self._paper_mode},
        )

    async def get_positions(self) -> list[Position]:
        data = await self._rest.get_positions()
        out: list[Position] = []
        # Kalshi /portfolio/positions returns market_positions: [{ticker, position, ...}].
        # 'position' is a signed integer of YES contracts (negative => NO exposure).
        for mp in data.get("market_positions", []) or []:
            ticker = mp.get("ticker")
            net = int(mp.get("position", 0))
            if ticker is None or net == 0:
                continue
            outcome = "YES" if net > 0 else "NO"
            size = Decimal(abs(net))
            avg_cents = mp.get("market_exposure")  # cents
            avg_prob = (
                cents_to_prob(int(avg_cents) // max(abs(net), 1))
                if avg_cents is not None
                else Decimal("0.5")
            )
            out.append(
                Position(
                    market_id=ticker,
                    venue="kalshi",
                    outcome_id=outcome,
                    size=size,
                    avg_price_prob=avg_prob,
                    unrealized_pnl=Decimal(str(mp.get("realized_pnl", 0))) / Decimal(100),
                    as_of_ts=time.time_ns(),
                    native=mp,
                )
            )
        # Also include paper open-position derived snapshots so risk state can rebuild
        # from a paper run if needed. Only positions that have any filled qty.
        if self._paper_mode:
            paper_filled: dict[tuple[str, str, Side], Decimal] = {}
            for o in self._paper.open_orders():
                if o.filled <= 0:
                    continue
                key = (o.market_id, o.outcome_id, o.side)
                paper_filled[key] = paper_filled.get(key, Decimal("0")) + o.filled
            for (market_id, outcome_id, side), qty in paper_filled.items():
                out.append(
                    Position(
                        market_id=market_id,
                        venue="kalshi",
                        outcome_id=outcome_id,
                        size=qty if side == Side.BUY else -qty,
                        avg_price_prob=Decimal("0.5"),
                        unrealized_pnl=Decimal("0"),
                        as_of_ts=time.time_ns(),
                        native={"paper": True},
                    )
                )
        return out

    # ==================================================================
    # Order lifecycle
    # ==================================================================

    async def place_limit(
        self,
        market_id: str,
        side: Side,
        size: Decimal,
        price: Decimal,
        tif: TIF,
        *,
        outcome_id: str = "YES",
    ) -> str:
        canon = canonicalize_outcome(outcome_id)
        # Validate price up-front so paper and live raise the same canonical error.
        try:
            cents = prob_to_cents(price)
        except ValueError as exc:
            raise InvalidPrice(str(exc)) from exc

        if self._paper_mode:
            order_id = self._paper.place_limit(
                market_id=market_id,
                outcome_id=canon,
                side=side,
                size=Decimal(str(size)),
                price_prob=Decimal(str(price)),
                tif=tif,
            )
            self._ensure_paper_poller(market_id)
            log.info(
                "kalshi.paper.place_limit",
                order_id=order_id,
                market_id=market_id,
                outcome_id=canon,
                side=side.value,
                size=str(size),
                price_prob=str(price),
                cents=cents,
                tif=tif.value,
            )
            return order_id

        body = {
            "ticker": market_id,
            "client_order_id": f"exec-{int(time.time_ns())}",
            "type": "limit",
            "action": side_to_action(side),
            "side": to_native_outcome(canon),
            "count": int(size),
            ("yes_price" if canon == "YES" else "no_price"): cents,
            "time_in_force": _kalshi_tif(tif),
        }
        resp = await self._rest.create_order(body)
        order = resp.get("order") or resp
        return str(order.get("order_id"))

    async def place_market(
        self,
        market_id: str,
        side: Side,
        size: Decimal,
        *,
        outcome_id: str = "YES",
    ) -> str:
        canon = canonicalize_outcome(outcome_id)
        if self._paper_mode:
            order_id = self._paper.place_market(
                market_id=market_id,
                outcome_id=canon,
                side=side,
                size=Decimal(str(size)),
            )
            self._ensure_paper_poller(market_id)
            log.info(
                "kalshi.paper.place_market",
                order_id=order_id,
                market_id=market_id,
                outcome_id=canon,
                side=side.value,
                size=str(size),
            )
            return order_id

        body = {
            "ticker": market_id,
            "client_order_id": f"exec-{int(time.time_ns())}",
            "type": "market",
            "action": side_to_action(side),
            "side": to_native_outcome(canon),
            "count": int(size),
        }
        resp = await self._rest.create_order(body)
        order = resp.get("order") or resp
        return str(order.get("order_id"))

    async def place_stop(
        self,
        market_id: str,
        side: Side,
        size: Decimal,
        trigger_price: Decimal,
    ) -> str:
        raise NotSupportedError("Kalshi does not offer native stop orders")

    async def replace_order(
        self,
        order_id: str,
        new_price: Decimal | None,
        new_size: Decimal | None,
    ) -> str:
        if self._paper_mode:
            return self._paper.replace(order_id, new_price, new_size)

        body: dict[str, Any] = {}
        if new_price is not None:
            try:
                body["price"] = prob_to_cents(new_price)
            except ValueError as exc:
                raise InvalidPrice(str(exc)) from exc
        if new_size is not None:
            body["count"] = int(new_size)
        resp = await self._rest.amend_order(order_id, body)
        order = resp.get("order") or resp
        return str(order.get("order_id", order_id))

    async def cancel_order(self, order_id: str) -> bool:
        if self._paper_mode:
            return self._paper.cancel(order_id)
        try:
            await self._rest.cancel_order(order_id)
            return True
        except Exception as exc:  # pragma: no cover — error mapping covers shape
            log.warning("kalshi.cancel_order.failed", order_id=order_id, error=str(exc))
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        if self._paper_mode and order_id.startswith("paper-"):
            return self._paper.status(order_id)
        data = await self._rest.get_order(order_id)
        raw = data.get("order") or data
        return _orderstatus_from_raw(raw)

    async def get_fills(self, since_ts: int) -> list[Fill]:
        # since_ts is nanoseconds per the spec. Kalshi accepts seconds via
        # min_ts; we convert here.
        params: dict[str, Any] = {}
        if since_ts:
            params["min_ts"] = int(since_ts // 1_000_000_000)
        live: list[Fill] = []
        try:
            data = await self._rest.get_fills(**params)
            for raw in data.get("fills", []) or []:
                live.append(_fill_from_raw(raw))
        except Exception as exc:
            log.warning("kalshi.get_fills.failed", error=str(exc))
        if self._paper_mode:
            return live + self._paper.fills_since(since_ts)
        return live

    # ==================================================================
    # Subscriptions (always live against Kalshi)
    # ==================================================================

    async def _ensure_ws(self) -> KalshiWS:
        if self._ws is None:
            if self._auth is None:
                self._auth = auth_from_env()
            self._ws = KalshiWS(self._auth, url=self._ws_url, ws_path=DEFAULT_WS_PATH)
            await self._ws.start()
        return self._ws

    async def subscribe_orderbook(self, market_ids: Iterable[str]) -> AsyncIterator[OrderbookEvent]:
        markets = tuple(market_ids)
        ws = await self._ensure_ws()
        sub = await ws.subscribe(["orderbook_delta"], markets)
        # Track per-market snapshot state so we can yield canonical OrderbookEvents.
        snapshots: dict[str, dict[str, list[list[int]]]] = {}
        try:
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                msg = item
                body = msg.get("msg") or {}
                ticker = body.get("market_ticker")
                if not ticker:
                    continue
                if msg.get("type") == "orderbook_snapshot":
                    snapshots[ticker] = {
                        "yes": [list(x) for x in (body.get("yes") or [])],
                        "no": [list(x) for x in (body.get("no") or [])],
                    }
                elif msg.get("type") == "orderbook_delta":
                    snap = snapshots.setdefault(ticker, {"yes": [], "no": []})
                    side = body.get("side")  # "yes" or "no"
                    price = int(body.get("price", 0))
                    delta = int(body.get("delta", 0))
                    if side in ("yes", "no") and price:
                        levels = snap[side]
                        for lvl in levels:
                            if lvl[0] == price:
                                lvl[1] = max(0, int(lvl[1]) + delta)
                                break
                        else:
                            if delta > 0:
                                levels.append([price, delta])
                        snap[side] = [lvl for lvl in levels if lvl[1] > 0]
                else:
                    continue
                ob = parse_orderbook(ticker, snapshots[ticker], outcome="YES")
                yield OrderbookEvent(orderbook=ob)
        finally:
            await ws.unsubscribe(sub.sub_id)

    async def subscribe_trades(self, market_ids: Iterable[str]) -> AsyncIterator[TradeEvent]:
        markets = tuple(market_ids)
        ws = await self._ensure_ws()
        sub = await ws.subscribe(["trade"], markets)
        try:
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                body = item.get("msg") or {}
                ticker = body.get("market_ticker")
                if not ticker:
                    continue
                # Kalshi trade msg: {market_ticker, yes_price, no_price, count, taker_side, ts}
                cents = int(body.get("yes_price", 0)) or (100 - int(body.get("no_price", 0)))
                taker = body.get("taker_side", "yes").lower()
                outcome = "YES" if taker == "yes" else "NO"
                yield TradeEvent(
                    venue="kalshi",
                    market_id=ticker,
                    outcome_id=outcome,
                    side=Side.BUY,
                    size=Decimal(str(body.get("count", 0))),
                    price_prob=cents_to_prob(cents),
                    ts_ns=int(body.get("ts", time.time())) * 1_000_000_000,
                    native=body,
                )
        finally:
            await ws.unsubscribe(sub.sub_id)

    async def subscribe_fills(self) -> AsyncIterator[FillEvent]:
        # Paper fills come from the in-process queue. Live fills come from the
        # Kalshi 'fill' channel. We multiplex both so a single subscriber sees
        # everything.
        live_iter: AsyncIterator[FillEvent] | None = None
        if not self._paper_mode:
            ws = await self._ensure_ws()
            sub = await ws.subscribe(["fill"], None)

            async def _live() -> AsyncIterator[FillEvent]:
                try:
                    while True:
                        item = await sub.queue.get()
                        if item is None:
                            return
                        if isinstance(item, Exception):
                            raise item
                        body = item.get("msg") or {}
                        yield FillEvent(fill=_fill_from_raw(body))
                finally:
                    await ws.unsubscribe(sub.sub_id)

            live_iter = _live()

        paper_q: asyncio.Queue[Fill] = asyncio.Queue(maxsize=10_000)
        if self._paper_mode:
            self._paper_fill_queues.append(paper_q)

        try:
            if live_iter is not None:
                async for ev in live_iter:
                    yield ev
            else:
                while True:
                    fill = await paper_q.get()
                    yield FillEvent(fill=fill)
        finally:
            if paper_q in self._paper_fill_queues:
                self._paper_fill_queues.remove(paper_q)

    async def subscribe_positions(self) -> AsyncIterator[PositionEvent]:
        if self._paper_mode:
            # Paper positions are derived; emit a single snapshot then idle.
            for p in await self.get_positions():
                yield PositionEvent(position=p)
            # Park forever (consumer cancels the iterator).
            await asyncio.Event().wait()
            return  # pragma: no cover
        ws = await self._ensure_ws()
        sub = await ws.subscribe(["position"], None)
        try:
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                body = item.get("msg") or {}
                ticker = body.get("market_ticker") or body.get("ticker")
                net = int(body.get("position", 0))
                if not ticker:
                    continue
                outcome = "YES" if net >= 0 else "NO"
                yield PositionEvent(
                    position=Position(
                        market_id=ticker,
                        venue="kalshi",
                        outcome_id=outcome,
                        size=Decimal(abs(net)),
                        avg_price_prob=Decimal("0.5"),
                        unrealized_pnl=Decimal("0"),
                        as_of_ts=time.time_ns(),
                        native=body,
                    )
                )
        finally:
            await ws.unsubscribe(sub.sub_id)

    # ==================================================================
    # Paper fill simulation
    # ==================================================================

    def _ensure_paper_poller(self, market_id: str) -> None:
        if not self._paper_mode:
            return
        if market_id in self._paper_pollers:
            return
        loop = asyncio.get_event_loop()
        task = loop.create_task(
            self._paper_poll_loop(market_id),
            name=f"kalshi-paper-poll-{market_id}",
        )
        self._paper_pollers[market_id] = task

    async def _paper_poll_loop(self, market_id: str) -> None:
        log.info("kalshi.paper.poller.start", market_id=market_id)
        try:
            while not self._closed:
                try:
                    ob = await self.get_orderbook(market_id)
                    new_fills = self._paper.try_fill_against(ob)
                    for f in new_fills:
                        log.info(
                            "kalshi.paper.fill",
                            order_id=f.order_id,
                            market_id=f.market_id,
                            outcome_id=f.outcome_id,
                            side=f.side.value,
                            size=str(f.size),
                            price_prob=str(f.price_prob),
                        )
                        for q in list(self._paper_fill_queues):
                            try:
                                q.put_nowait(f)
                            except asyncio.QueueFull:
                                pass
                    # Stop polling if no open paper orders remain on this market.
                    if not self._paper.open_orders(market_id=market_id):
                        log.debug("kalshi.paper.poller.idle_exit", market_id=market_id)
                        del self._paper_pollers[market_id]
                        return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("kalshi.paper.poller.error", market_id=market_id, error=str(exc))
                await asyncio.sleep(self._paper_fill_poll_sec)
        except asyncio.CancelledError:
            log.info("kalshi.paper.poller.cancelled", market_id=market_id)
            raise

    # Test helper — run one fill cycle synchronously without the poller.
    async def _paper_tick(self, market_id: str) -> list[Fill]:
        ob = await self.get_orderbook(market_id)
        new_fills = self._paper.try_fill_against(ob)
        for f in new_fills:
            for q in list(self._paper_fill_queues):
                try:
                    q.put_nowait(f)
                except asyncio.QueueFull:
                    pass
        return new_fills


# ---------------------------------------------------------------------------
# Helpers for parsing Kalshi raw shapes
# ---------------------------------------------------------------------------


def _market_from_raw(raw: dict[str, Any]) -> Market:
    ticker = raw.get("ticker") or raw.get("market_ticker")
    event_id = raw.get("event_ticker")
    title = raw.get("title") or raw.get("yes_sub_title") or ticker or ""
    status = raw.get("status") or "unknown"
    close_iso = raw.get("close_time")
    close_ns = None
    if isinstance(close_iso, str):
        try:
            from datetime import datetime
            close_ns = int(datetime.fromisoformat(close_iso.replace("Z", "+00:00")).timestamp() * 1_000_000_000)
        except Exception:
            close_ns = None
    return Market(
        market_id=str(ticker),
        venue="kalshi",
        event_id=event_id,
        title=str(title),
        status=str(status).upper(),
        close_ts=close_ns,
        outcomes=("YES", "NO"),
        native=raw,
    )


def _kalshi_tif(tif: TIF) -> str:
    # Kalshi accepts 'GTC', 'IOC', 'FOK' (no DAY in v2 — we map DAY to GTC and stash original in metadata if needed).
    if tif == TIF.IOC:
        return "IOC"
    if tif == TIF.FOK:
        return "FOK"
    return "GTC"


def _orderstatus_from_raw(raw: dict[str, Any]) -> OrderStatus:
    state_map = {
        "resting": OrderState.OPEN,
        "open": OrderState.OPEN,
        "executed": OrderState.FILLED,
        "filled": OrderState.FILLED,
        "canceled": OrderState.CANCELLED,
        "cancelled": OrderState.CANCELLED,
        "expired": OrderState.EXPIRED,
        "rejected": OrderState.REJECTED,
        "pending": OrderState.PENDING,
    }
    state = state_map.get(str(raw.get("status", "")).lower(), OrderState.OPEN)
    side = Side.BUY if str(raw.get("action", "buy")).lower() == "buy" else Side.SELL
    outcome = canonicalize_outcome(raw.get("side") or raw.get("outcome") or "yes")
    cents = raw.get("yes_price") if outcome == "YES" else raw.get("no_price")
    price = cents_to_prob(int(cents)) if cents is not None else None
    placed_iso = raw.get("created_time") or raw.get("placed_time")
    placed_ns = _iso_to_ns(placed_iso) if placed_iso else time.time_ns()
    updated_iso = raw.get("last_update_time")
    updated_ns = _iso_to_ns(updated_iso) if updated_iso else placed_ns
    return OrderStatus(
        order_id=str(raw.get("order_id")),
        venue="kalshi",
        market_id=str(raw.get("ticker") or raw.get("market_ticker")),
        outcome_id=outcome,
        side=side,
        state=state,
        size=Decimal(str(raw.get("count", 0))),
        filled=Decimal(str(raw.get("filled_count", 0) or 0)),
        price_prob=price,
        tif=None,
        placed_ts_ns=placed_ns,
        last_update_ts_ns=updated_ns,
        native=raw,
    )


def _fill_from_raw(raw: dict[str, Any]) -> Fill:
    outcome = canonicalize_outcome(raw.get("side") or raw.get("outcome") or "yes")
    cents = raw.get("yes_price") if outcome == "YES" else raw.get("no_price")
    price = cents_to_prob(int(cents)) if cents is not None else Decimal("0.5")
    side = Side.BUY if str(raw.get("action", "buy")).lower() == "buy" else Side.SELL
    ts_ns = _iso_to_ns(raw.get("created_time")) if raw.get("created_time") else int(time.time_ns())
    return Fill(
        fill_id=str(raw.get("trade_id") or raw.get("fill_id") or f"k-{int(time.time_ns())}"),
        order_id=str(raw.get("order_id")),
        venue="kalshi",
        market_id=str(raw.get("ticker") or raw.get("market_ticker")),
        outcome_id=outcome,
        side=side,
        size=Decimal(str(raw.get("count", 0))),
        price_prob=price,
        fee=(Decimal(str(raw.get("fee", 0))) / Decimal(100)) if raw.get("fee") is not None else Decimal("0"),
        ts_ns=ts_ns,
        native_quote={"cents": int(cents) if cents is not None else None},
        native=raw,
    )


def _iso_to_ns(s: str) -> int:
    try:
        from datetime import datetime
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1_000_000_000)
    except Exception:
        return time.time_ns()
