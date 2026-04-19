"""
Kalshi WebSocket client.

Single shared connection; multiple subscribe_* calls multiplex over it via
a per-channel asyncio.Queue. Reconnects with exponential backoff. Errors on
the connection are surfaced through the per-subscription queues as canonical
VenueError instances using errors.map_ws_error so the consumer can decide
to crash, pause, or retry.

Channels supported:
    orderbook_delta  -> orderbook snapshot + incremental deltas
    trade            -> public trades
    fill             -> private fills (authed)
    position         -> private position updates (Kalshi names this 'position'
                       on the v2 trade WS; see Kalshi docs for spec drift)
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import websockets

from ...core.logging import get_logger
from .auth import KalshiAuth
from .errors import map_ws_error


log = get_logger("executor.venue.kalshi.ws")


DEFAULT_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
DEFAULT_WS_PATH = "/trade-api/ws/v2"


@dataclass
class _ChannelSub:
    """A logical subscription — one queue per (channel, market_filter)."""
    sub_id: int
    channels: tuple[str, ...]
    market_tickers: tuple[str, ...]   # empty == all
    queue: asyncio.Queue[Any] = field(default_factory=lambda: asyncio.Queue(maxsize=10_000))
    closed: bool = False


class KalshiWS:
    def __init__(
        self,
        auth: KalshiAuth,
        *,
        url: str = DEFAULT_WS_URL,
        ws_path: str = DEFAULT_WS_PATH,
        ping_interval: float = 10.0,
        ping_timeout: float = 10.0,
    ) -> None:
        self._auth = auth
        self._url = url
        self._ws_path = ws_path
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout

        self._subs: dict[int, _ChannelSub] = {}
        self._next_id = 1
        self._ws: Any = None
        self._connector_task: asyncio.Task[None] | None = None
        self._running = False
        self._lock = asyncio.Lock()
        self._connected = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._connector_task = asyncio.create_task(self._connector(), name="kalshi-ws")

    async def stop(self) -> None:
        self._running = False
        if self._connector_task is not None:
            self._connector_task.cancel()
            try:
                await self._connector_task
            except (asyncio.CancelledError, Exception):
                pass
            self._connector_task = None
        for sub in list(self._subs.values()):
            sub.closed = True
            try:
                sub.queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        channels: Iterable[str],
        market_tickers: Iterable[str] | None = None,
    ) -> _ChannelSub:
        async with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            sub = _ChannelSub(
                sub_id=sub_id,
                channels=tuple(channels),
                market_tickers=tuple(market_tickers or ()),
            )
            self._subs[sub_id] = sub

        # Send subscribe frame if connection is up; otherwise the connector
        # will replay all subs after reconnect.
        if self._ws is not None:
            try:
                await self._send_subscribe(sub)
            except Exception as exc:
                log.warning("kalshi.ws.subscribe.send_failed", error=str(exc))
        return sub

    async def unsubscribe(self, sub_id: int) -> None:
        async with self._lock:
            sub = self._subs.pop(sub_id, None)
        if sub is not None:
            sub.closed = True
            try:
                sub.queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    # ------------------------------------------------------------------
    # Connector / pump
    # ------------------------------------------------------------------

    async def _connector(self) -> None:
        delay = 1.0
        while self._running:
            try:
                headers = self._auth.ws_headers(self._ws_path)
                log.info("kalshi.ws.connect", url=self._url)
                async with websockets.connect(
                    self._url,
                    additional_headers=headers,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    close_timeout=5,
                    max_size=2**24,
                ) as ws:
                    self._ws = ws
                    self._connected.set()
                    delay = 1.0
                    # Replay all current subscriptions.
                    async with self._lock:
                        subs = list(self._subs.values())
                    for sub in subs:
                        try:
                            await self._send_subscribe(sub)
                        except Exception as exc:
                            log.warning("kalshi.ws.replay_subscribe.failed", error=str(exc))
                    await self._read_loop(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("kalshi.ws.connection_lost", error=str(exc))
            finally:
                self._ws = None
                self._connected.clear()
            if not self._running:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

    async def _send_subscribe(self, sub: _ChannelSub) -> None:
        if self._ws is None:
            return
        params: dict[str, Any] = {"channels": list(sub.channels)}
        if sub.market_tickers:
            params["market_tickers"] = list(sub.market_tickers)
        frame = {"id": sub.sub_id, "cmd": "subscribe", "params": params}
        await self._ws.send(json.dumps(frame))
        log.debug("kalshi.ws.subscribe.sent", sub_id=sub.sub_id, channels=sub.channels)

    async def _read_loop(self, ws: Any) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                log.warning("kalshi.ws.parse_failed", raw=str(raw)[:200])
                continue
            mtype = msg.get("type")
            if mtype == "error":
                err = map_ws_error(msg)
                log.warning("kalshi.ws.error_frame", msg=msg)
                # Fan to every sub so consumers can decide to give up.
                async with self._lock:
                    subs = list(self._subs.values())
                for sub in subs:
                    try:
                        sub.queue.put_nowait(err)
                    except asyncio.QueueFull:
                        pass
                continue
            # Dispatch to subs whose channel set claims this msg type.
            await self._dispatch(msg)

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        mtype = msg.get("type", "")
        ticker = None
        body = msg.get("msg")
        if isinstance(body, dict):
            ticker = body.get("market_ticker")
        async with self._lock:
            subs = list(self._subs.values())
        for sub in subs:
            if sub.closed:
                continue
            # Channel match: orderbook_delta covers snapshot+delta+lifecycle frames.
            if not _channel_matches(sub.channels, mtype):
                continue
            if sub.market_tickers and ticker is not None and ticker not in sub.market_tickers:
                continue
            try:
                sub.queue.put_nowait(msg)
            except asyncio.QueueFull:
                log.warning("kalshi.ws.subscriber.drop", sub_id=sub.sub_id)


def _channel_matches(channels: tuple[str, ...], mtype: str) -> bool:
    """Map a Kalshi WS message type to the channel name a subscriber asked for."""
    if not mtype:
        return False
    chset = set(channels)
    # orderbook_delta channel emits both snapshots and deltas.
    if mtype in ("orderbook_snapshot", "orderbook_delta") and "orderbook_delta" in chset:
        return True
    if mtype == "trade" and "trade" in chset:
        return True
    if mtype == "fill" and "fill" in chset:
        return True
    if mtype in ("position", "position_update") and "position" in chset:
        return True
    return False
