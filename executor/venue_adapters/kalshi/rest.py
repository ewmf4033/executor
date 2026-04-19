"""
Kalshi REST client (async, aiohttp).

Thin wrapper around the v2 REST API. Adds KALSHI-ACCESS-* signature headers,
parses JSON, and converts non-2xx responses into canonical VenueError types
via errors.map_http_error.

This module is venue-flavored — callers get raw Kalshi dicts back. The
adapter (adapter.py) does the canonical-shape conversion.
"""
from __future__ import annotations

from typing import Any

import aiohttp

from ...core.logging import get_logger
from .auth import KalshiAuth
from .errors import map_http_error


log = get_logger("executor.venue.kalshi.rest")

DEFAULT_BASE_URL = "https://api.elections.kalshi.com"
API_PREFIX = "/trade-api/v2"


class KalshiREST:
    def __init__(
        self,
        auth: KalshiAuth | None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_sec: float = 10.0,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._auth = auth
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_sec)
        self._owned_session = session is None
        self._session: aiohttp.ClientSession | None = session

    async def __aenter__(self) -> "KalshiREST":
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owned_session = True
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owned_session and self._session is not None:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Core request
    # ------------------------------------------------------------------

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        authed: bool = True,
    ) -> dict[str, Any]:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owned_session = True

        full_path = path if path.startswith(API_PREFIX) else f"{API_PREFIX}{path}"
        url = f"{self._base_url}{full_path}"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if authed:
            if self._auth is None:
                raise RuntimeError(f"authed Kalshi call but no auth configured: {method} {full_path}")
            # Sign with the path only (no query string), per kalshi_auth contract.
            headers.update(self._auth.headers(method, full_path))

        log.debug(
            "kalshi.rest.request",
            method=method,
            path=full_path,
            authed=authed,
            params=params,
        )

        try:
            async with self._session.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=headers,
            ) as resp:
                text = await resp.text()
                status = resp.status
        except aiohttp.ClientError as exc:
            log.warning("kalshi.rest.transport_error", method=method, path=full_path, error=str(exc))
            from ...core.types import VenueDown
            raise VenueDown(f"kalshi transport: {exc}") from exc

        if status >= 400:
            try:
                import json
                body: Any = json.loads(text) if text else {}
            except Exception:
                body = text
            log.warning(
                "kalshi.rest.error",
                method=method,
                path=full_path,
                status=status,
                body=str(body)[:300],
            )
            raise map_http_error(status, body)

        if not text:
            return {}
        import json
        return json.loads(text)

    # ------------------------------------------------------------------
    # Convenience wrappers — used by adapter.py
    # ------------------------------------------------------------------

    async def get_markets(self, **params: Any) -> dict[str, Any]:
        return await self.request("GET", "/markets", params=params or None, authed=False)

    async def get_market(self, ticker: str) -> dict[str, Any]:
        return await self.request("GET", f"/markets/{ticker}", authed=False)

    async def get_orderbook(self, ticker: str) -> dict[str, Any]:
        return await self.request("GET", f"/markets/{ticker}/orderbook", authed=False)

    async def get_balance(self) -> dict[str, Any]:
        return await self.request("GET", "/portfolio/balance", authed=True)

    async def get_positions(self, **params: Any) -> dict[str, Any]:
        return await self.request("GET", "/portfolio/positions", params=params or None, authed=True)

    async def get_orders(self, **params: Any) -> dict[str, Any]:
        return await self.request("GET", "/portfolio/orders", params=params or None, authed=True)

    async def get_order(self, order_id: str) -> dict[str, Any]:
        return await self.request("GET", f"/portfolio/orders/{order_id}", authed=True)

    async def get_fills(self, **params: Any) -> dict[str, Any]:
        return await self.request("GET", "/portfolio/fills", params=params or None, authed=True)

    async def create_order(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self.request("POST", "/portfolio/orders", json_body=body, authed=True)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        return await self.request("DELETE", f"/portfolio/orders/{order_id}", authed=True)

    async def amend_order(self, order_id: str, body: dict[str, Any]) -> dict[str, Any]:
        # Kalshi exposes amend via POST /portfolio/orders/{id}/amend in v2.
        return await self.request("POST", f"/portfolio/orders/{order_id}/amend", json_body=body, authed=True)
