"""Tests for executor.tools.kalshi_rest_snapshot.

No live network. A fake ``Fetch`` is injected so URL/method assertions are
hermetic. The module's ``_assert_public_get`` guard is exercised both
directly (rejection unit tests) and indirectly (every fake-fetch invocation
goes through it before returning).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from executor.tools import kalshi_rest_snapshot as snap


# ---------------------------------------------------------------------------
# Fake fetch
# ---------------------------------------------------------------------------


class FakeFetch:
    """Records calls and returns canned responses keyed by URL path."""

    def __init__(self, responses: dict[str, tuple[int, object]] | None = None) -> None:
        self.responses = dict(responses or {})
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method: str, url: str, timeout_sec: float):
        # Mirror default_fetch: enforce the guard before "I/O".
        from urllib.parse import urlparse

        path = urlparse(url).path
        snap._assert_public_get(method, path)
        self.calls.append((method, url))
        if path in self.responses:
            status, body = self.responses[path]
            return status, body, 1.234
        # Default 404 to make missing fixtures loud.
        return 404, {"error": "not found"}, 1.0


def _markets_payload(tickers: list[str]) -> dict:
    return {
        "markets": [
            {"ticker": t, "status": "open", "yes_bid": 50, "yes_ask": 52}
            for t in tickers
        ],
        "cursor": "",
    }


def _orderbook_payload(ticker: str) -> dict:
    return {"orderbook": {"yes": [[50, 100]], "no": [[48, 200]]}, "ticker": ticker}


# ---------------------------------------------------------------------------
# 1. Public markets URL is GET-only and well-formed
# ---------------------------------------------------------------------------


def test_build_url_markets_is_public_path():
    url = snap.build_url("https://api.elections.kalshi.com", "/markets", {"status": "open"})
    assert url.startswith("https://api.elections.kalshi.com/trade-api/v2/markets")
    assert "status=open" in url
    # The default fetch will assert GET-only; verify the guard doesn't reject this path.
    snap._assert_public_get("GET", "/trade-api/v2/markets")


# ---------------------------------------------------------------------------
# 2-4. Forbidden path / method guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/trade-api/v2/portfolio/balance",
        "/trade-api/v2/portfolio/orders",
        "/trade-api/v2/portfolio/positions",
        "/trade-api/v2/portfolio/fills",
        "/trade-api/v2/portfolio/orders/abc/amend",
    ],
)
def test_forbidden_path_guard_rejects_portfolio_and_orders(path):
    with pytest.raises(ValueError, match="forbidden path segment"):
        snap._assert_public_get("GET", path)


def test_forbidden_path_guard_rejects_orders_in_any_position():
    # Even a hypothetical /v2/orders/foo path is rejected by the substring rule.
    with pytest.raises(ValueError, match="forbidden path segment"):
        snap._assert_public_get("GET", "/trade-api/v2/orders/foo")


@pytest.mark.parametrize("method", ["POST", "DELETE", "PUT", "PATCH", "HEAD"])
def test_forbidden_method_guard_rejects_non_get(method):
    with pytest.raises(ValueError, match="forbidden method"):
        snap._assert_public_get(method, "/trade-api/v2/markets")


# ---------------------------------------------------------------------------
# 5. Dry-run does not write output
# ---------------------------------------------------------------------------


def test_dry_run_does_not_write_output(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    fake = FakeFetch(
        {
            "/trade-api/v2/markets": (200, _markets_payload(["A", "B"])),
            "/trade-api/v2/markets/A/orderbook": (200, _orderbook_payload("A")),
            "/trade-api/v2/markets/B/orderbook": (200, _orderbook_payload("B")),
        }
    )
    rc = snap.run(
        ["--limit", "5", "--dry-run", "--out", str(out), "--sleep-sec", "0"],
        fetch=fake,
        sleep=lambda s: None,
    )
    assert rc == 0
    assert not out.exists()


# ---------------------------------------------------------------------------
# 6. JSONL records include captured timestamps
# ---------------------------------------------------------------------------


def test_jsonl_includes_captured_timestamps(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    fake = FakeFetch(
        {
            "/trade-api/v2/markets": (200, _markets_payload(["A"])),
            "/trade-api/v2/markets/A/orderbook": (200, _orderbook_payload("A")),
        }
    )
    rc = snap.run(
        ["--limit", "1", "--out", str(out), "--sleep-sec", "0"],
        fetch=fake,
        sleep=lambda s: None,
    )
    assert rc == 0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert "captured_wall_ts_ns" in rec
    assert "captured_monotonic_ns" in rec
    assert rec["source"] == "kalshi_rest_public"
    assert rec["ticker"] == "A"


# ---------------------------------------------------------------------------
# 7. Limit is respected
# ---------------------------------------------------------------------------


def test_limit_is_respected(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    tickers = [f"T{i}" for i in range(10)]
    responses = {"/trade-api/v2/markets": (200, _markets_payload(tickers))}
    for t in tickers:
        responses[f"/trade-api/v2/markets/{t}/orderbook"] = (200, _orderbook_payload(t))
    fake = FakeFetch(responses)
    rc = snap.run(
        ["--limit", "3", "--out", str(out), "--sleep-sec", "0"],
        fetch=fake,
        sleep=lambda s: None,
    )
    assert rc == 0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# 8. Series-prefix filter is respected
# ---------------------------------------------------------------------------


def test_series_prefix_filter(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    tickers = ["KXFED-1", "KXFED-2", "OTHER-1", "OTHER-2"]
    responses = {"/trade-api/v2/markets": (200, _markets_payload(tickers))}
    for t in tickers:
        responses[f"/trade-api/v2/markets/{t}/orderbook"] = (200, _orderbook_payload(t))
    fake = FakeFetch(responses)
    rc = snap.run(
        [
            "--limit",
            "10",
            "--series-prefix",
            "KXFED",
            "--out",
            str(out),
            "--sleep-sec",
            "0",
        ],
        fetch=fake,
        sleep=lambda s: None,
    )
    assert rc == 0
    captured_tickers = [json.loads(l)["ticker"] for l in out.read_text().strip().splitlines()]
    assert set(captured_tickers) == {"KXFED-1", "KXFED-2"}


# ---------------------------------------------------------------------------
# 9. Repeated --ticker skips discovery
# ---------------------------------------------------------------------------


def test_explicit_tickers_skip_discovery(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    fake = FakeFetch(
        {
            "/trade-api/v2/markets/X/orderbook": (200, _orderbook_payload("X")),
            "/trade-api/v2/markets/X": (200, {"market": {"ticker": "X", "status": "open"}}),
            "/trade-api/v2/markets/Y/orderbook": (200, _orderbook_payload("Y")),
            "/trade-api/v2/markets/Y": (200, {"market": {"ticker": "Y", "status": "open"}}),
        }
    )
    rc = snap.run(
        [
            "--ticker",
            "X",
            "--ticker",
            "Y",
            "--out",
            str(out),
            "--sleep-sec",
            "0",
        ],
        fetch=fake,
        sleep=lambda s: None,
    )
    assert rc == 0
    # No /markets discovery call should have been issued.
    discovery_calls = [c for c in fake.calls if c[1].endswith("/trade-api/v2/markets") or "/markets?" in c[1]]
    assert discovery_calls == []
    lines = out.read_text().strip().splitlines()
    assert [json.loads(l)["ticker"] for l in lines] == ["X", "Y"]


# ---------------------------------------------------------------------------
# 10. Per-orderbook failure records error and continues
# ---------------------------------------------------------------------------


def test_per_orderbook_failure_continues(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    fake = FakeFetch(
        {
            "/trade-api/v2/markets": (200, _markets_payload(["A", "B"])),
            "/trade-api/v2/markets/A/orderbook": (200, _orderbook_payload("A")),
            # B's orderbook intentionally absent → 404 from FakeFetch default
        }
    )
    rc = snap.run(
        ["--limit", "5", "--out", str(out), "--sleep-sec", "0"],
        fetch=fake,
        sleep=lambda s: None,
    )
    assert rc == 0
    lines = [json.loads(l) for l in out.read_text().strip().splitlines()]
    by_ticker = {r["ticker"]: r for r in lines}
    assert "A" in by_ticker and "B" in by_ticker
    assert "orderbook" in by_ticker["A"]
    assert "error" in by_ticker["B"]
    assert "orderbook" not in by_ticker["B"]


# ---------------------------------------------------------------------------
# 11. Initial markets fetch failure exits nonzero
# ---------------------------------------------------------------------------


def test_initial_markets_fetch_failure_exits_nonzero(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    fake = FakeFetch(
        {
            # /markets returns 500 — discovery must fail loudly.
            "/trade-api/v2/markets": (500, {"error": "server error"}),
        }
    )
    rc = snap.run(
        ["--limit", "5", "--out", str(out), "--sleep-sec", "0"],
        fetch=fake,
        sleep=lambda s: None,
    )
    assert rc != 0
    assert not out.exists()


# ---------------------------------------------------------------------------
# 12. No Kalshi credentials are read
# ---------------------------------------------------------------------------


def test_does_not_require_kalshi_credentials(tmp_path: Path, monkeypatch):
    out = tmp_path / "out.jsonl"
    # Strip credentials from env to prove the tool never touches them.
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)

    fake = FakeFetch(
        {
            "/trade-api/v2/markets": (200, _markets_payload(["A"])),
            "/trade-api/v2/markets/A/orderbook": (200, _orderbook_payload("A")),
        }
    )
    rc = snap.run(
        ["--limit", "1", "--out", str(out), "--sleep-sec", "0"],
        fetch=fake,
        sleep=lambda s: None,
    )
    assert rc == 0
    assert out.exists()


# ---------------------------------------------------------------------------
# 13. Default output path is under audit-logs/market_snapshots/kalshi
# ---------------------------------------------------------------------------


def test_default_output_path_under_audit_logs_market_snapshots():
    p = snap.default_output_path()
    expected_parent = Path("/root/executor/audit-logs/market_snapshots/kalshi")
    assert p.parent == expected_parent
    assert p.suffix == ".jsonl"
    assert p.name.count("-") == 2  # YYYY-MM-DD.jsonl


# ---------------------------------------------------------------------------
# 14. No forbidden path is hit during normal capture
# ---------------------------------------------------------------------------


def test_no_forbidden_path_during_normal_capture(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    fake = FakeFetch(
        {
            "/trade-api/v2/markets": (200, _markets_payload(["A", "B"])),
            "/trade-api/v2/markets/A/orderbook": (200, _orderbook_payload("A")),
            "/trade-api/v2/markets/B/orderbook": (200, _orderbook_payload("B")),
        }
    )
    rc = snap.run(
        ["--limit", "5", "--out", str(out), "--sleep-sec", "0"],
        fetch=fake,
        sleep=lambda s: None,
    )
    assert rc == 0
    for method, url in fake.calls:
        assert method == "GET"
        for forbidden in snap.FORBIDDEN_PATH_PARTS:
            assert forbidden not in url, f"forbidden segment {forbidden!r} hit: {url}"
