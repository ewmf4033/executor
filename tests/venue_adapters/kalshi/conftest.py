"""
Shared fixtures for Kalshi adapter tests.

Live tests hit api.elections.kalshi.com read-only and are gated on
KALSHI_PRIVATE_KEY_PATH being present (auth isn't required for /markets
or /markets/{ticker}/orderbook, but the adapter wires an auth instance
regardless — get_account and get_positions would need it).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _live_available() -> bool:
    return Path("/root/kalshi_sports.key").exists() and bool(
        os.environ.get("KALSHI_API_KEY_ID") or True  # key_id is a constant we pass in env below
    )


live_only = pytest.mark.skipif(
    not Path("/root/kalshi_sports.key").exists(),
    reason="live Kalshi key not present",
)


@pytest.fixture
def kalshi_env(monkeypatch):
    """Set the Kalshi credentials the spec mandates for this droplet."""
    monkeypatch.setenv("KALSHI_API_KEY_ID", "02861f16-f9de-4874-b202-93c4746ab3eb")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", "/root/kalshi_sports.key")
    monkeypatch.setenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com")
    monkeypatch.setenv("KALSHI_WS_URL", "wss://api.elections.kalshi.com/trade-api/ws/v2")
