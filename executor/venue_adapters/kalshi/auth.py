"""
Kalshi RSA-PSS-SHA256 request signer.

Copied verbatim from /root/kalshi_lag_probe/kalshi_auth.py per the executor
spec (/root/trading-wiki/specs/0d-executor.md). Kept as a copy, not an
import, so the executor's failure modes are independent of the probe and
the probe's MASTA dependency.

Signs (timestamp_ms + method.upper() + path_without_query) with PSS+SHA256
and base64-encodes the resulting signature. Three KALSHI-ACCESS-* headers
are produced per request; for WebSocket handshakes the method is GET and
the path is the WS upgrade path (e.g. /trade-api/ws/v2).
"""
from __future__ import annotations

import base64
import os
import time
from urllib.parse import urlparse

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiAuth:
    """Builds signed headers for both REST and WebSocket handshakes."""

    def __init__(self, api_key_id: str, private_key_path: str):
        self.api_key_id = api_key_id
        self.private_key = self._load_private_key(private_key_path)

    @staticmethod
    def _load_private_key(key_path: str):
        with open(key_path, "rb") as handle:
            return serialization.load_pem_private_key(
                handle.read(), password=None, backend=default_backend()
            )

    def _create_signature(self, timestamp_ms: str, method: str, path: str) -> str:
        path_without_query = path.split("?")[0]
        message = f"{timestamp_ms}{method.upper()}{path_without_query}".encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def sign(self, method: str, path: str, *, timestamp_ms: str | None = None) -> tuple[str, str]:
        """Return (timestamp_ms, signature_b64) for a given REST path. Test-friendly."""
        ts = timestamp_ms if timestamp_ms is not None else str(int(time.time() * 1000))
        return ts, self._create_signature(ts, method, path)

    def headers(self, method: str, url_or_path: str) -> dict[str, str]:
        """Three KALSHI-ACCESS-* headers plus Content-Type for a REST or WS request."""
        timestamp_ms = str(int(time.time() * 1000))
        if url_or_path.startswith("http") or url_or_path.startswith("ws"):
            path = urlparse(url_or_path).path
        else:
            path = url_or_path
        signature = self._create_signature(timestamp_ms, method, path)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "Content-Type": "application/json",
        }

    def ws_headers(self, ws_path: str = "/trade-api/ws/v2") -> dict[str, str]:
        """Headers for a WebSocket handshake — method is GET for the upgrade."""
        return self.headers("GET", ws_path)


def auth_from_env() -> KalshiAuth:
    """
    Construct a KalshiAuth from KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_PATH env vars.

    Unlike the probe's auth_from_env, this does NOT shallow-parse a MASTA .env
    file — the executor reads its own environment and nothing more.
    """
    api_key_id = os.environ.get("KALSHI_API_KEY_ID")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if not api_key_id:
        raise RuntimeError("KALSHI_API_KEY_ID not set in environment")
    if not key_path:
        raise RuntimeError("KALSHI_PRIVATE_KEY_PATH not set in environment")
    return KalshiAuth(api_key_id=api_key_id, private_key_path=key_path)
