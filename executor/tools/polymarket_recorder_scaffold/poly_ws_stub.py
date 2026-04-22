"""
Polymarket WS recorder — SCAFFOLD, not operational.

Mirrors the connection/reconnect/gap-detection structure from the 0a1 Kalshi
recorder so that when a real implementation is wired, the shape is familiar.
All network-facing methods raise NotImplementedError. See README.md in this
directory for the delta-from-Kalshi notes.

Do NOT import this from the executor daemon, strategies, or any running
service. It exists as a reference shell only.

Phase 4.10 (0a2+ scaffold).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


POLY_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
# Target: match 0a1 recorder's reconnect schedule. Kalshi uses exponential
# backoff capped at 30s; Polymarket has no rate-limit docs on reconnect, so
# adopt the same envelope conservatively.
RECONNECT_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
# Polymarket expects a client `ping` every ~30s to keep the connection alive.
PING_INTERVAL_SECONDS = 25.0


@dataclass
class PolyBookMsg:
    """Normalized shape for a `book` (full orderbook) message."""

    ts_ns: int
    asset_id: str
    condition_id: str
    market_key: str                # canonical key shared across venues
    bids: list[tuple[float, float]]  # (price, size)
    asks: list[tuple[float, float]]
    tick_size: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolyTradeMsg:
    ts_ns: int
    asset_id: str
    condition_id: str
    market_key: str
    price: float
    size: float
    side: str  # "BUY" or "SELL" from the taker's perspective
    maker_orders: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class PolyWSRecorderStub:
    """Scaffold: shape matches 0a1 recorder. Methods stubbed."""

    def __init__(self, asset_ids: list[str], *, parquet_sink: Any = None) -> None:
        self._asset_ids = list(asset_ids)
        self._parquet_sink = parquet_sink
        self._last_msg_ts_ns: int = 0
        self._connected = False
        self._reconnects = 0

    async def connect(self) -> None:
        """Open the WS connection. Not implemented in scaffold."""
        raise NotImplementedError(
            "poly_ws_stub.connect: scaffold only. Implement against "
            "websockets/aiohttp when 0a2 is prioritized."
        )

    async def subscribe(self) -> None:
        """Send the initial subscription payload.

        Payload shape (per Polymarket docs):

            {"assets_ids": [...], "type": "market"}
        """
        raise NotImplementedError

    async def recv_loop(self) -> None:
        """Receive messages, normalize, and write to parquet_sink.

        On reconnect, Polymarket sends a fresh `book` first; the loop must
        treat that as authoritative and drop any stale `price_change`
        queued from the prior connection (gap detection).
        """
        raise NotImplementedError

    async def handle_book(self, msg: dict[str, Any]) -> PolyBookMsg:
        """Normalize a raw `book` message."""
        raise NotImplementedError

    async def handle_price_change(self, msg: dict[str, Any]) -> None:
        """Apply a delta to the in-memory book. Scaffold: no-op."""
        raise NotImplementedError

    async def handle_tick_size_change(self, msg: dict[str, Any]) -> None:
        """Polymarket-specific: tick size can change mid-life. Must
        re-bucket orderbook history or storage will misalign."""
        raise NotImplementedError

    async def handle_trade(self, msg: dict[str, Any]) -> PolyTradeMsg:
        raise NotImplementedError

    async def reconnect_loop(self) -> None:
        """Exponential backoff with a cap. Shape matches 0a1."""
        raise NotImplementedError


# CLI entrypoint deliberately omitted — a scaffold with a CLI invites
# someone to run it. Re-add when the real implementation lands.
