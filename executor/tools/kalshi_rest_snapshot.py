"""
Phase 5a.0 — public Kalshi REST snapshot tool.

Manual, read-only observation of public Kalshi market data. Does not
authenticate, does not place/cancel/amend orders, does not read any
portfolio endpoint, does not subscribe to a WebSocket. Output is
append-only JSONL for passive analysis.

Hard safety guards (see ``FORBIDDEN_PATH_PARTS`` and ``_assert_public_get``):

- only ``GET`` requests are permitted
- request paths matching any forbidden substring are rejected before the
  network call (``/portfolio/``, ``/orders``, ``/fills``, ``/positions``,
  ``/balance``, ``/amend``)
- no ``KalshiAuth`` is constructed; ``KALSHI_API_KEY_ID`` and
  ``KALSHI_PRIVATE_KEY_PATH`` are not read.

Run as a module::

    python3 -m executor.tools.kalshi_rest_snapshot --status open --limit 20 --dry-run

See ``trading-wiki/ops/kalshi_rest_snapshot.md`` for runbook usage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

DEFAULT_BASE_URL = "https://api.elections.kalshi.com"
API_PREFIX = "/trade-api/v2"
DEFAULT_OUTPUT_DIR = Path("/root/executor/audit-logs/market_snapshots/kalshi")
SOURCE_TAG = "kalshi_rest_public"

ALLOWED_METHOD = "GET"
FORBIDDEN_PATH_PARTS: tuple[str, ...] = (
    "/portfolio/",
    "/orders",
    "/fills",
    "/positions",
    "/balance",
    "/amend",
)


# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------


def _assert_public_get(method: str, path: str) -> None:
    """Reject any non-GET method or path pointing at a private endpoint.

    This is the single chokepoint between argparse-driven request building
    and the network. All ``Fetch`` implementations call it before doing I/O.
    """
    if method != ALLOWED_METHOD:
        raise ValueError(
            f"forbidden method {method!r}: kalshi_rest_snapshot is GET-only"
        )
    for part in FORBIDDEN_PATH_PARTS:
        if part in path:
            raise ValueError(
                f"forbidden path segment {part!r} in {path!r}: "
                "this tool only accesses public market data"
            )


# ---------------------------------------------------------------------------
# URL / payload helpers (pure)
# ---------------------------------------------------------------------------


def build_url(base_url: str, path: str, params: dict[str, Any] | None = None) -> str:
    """Compose a full URL under ``API_PREFIX`` from a relative path."""
    base = base_url.rstrip("/")
    full_path = path if path.startswith(API_PREFIX) else f"{API_PREFIX}{path}"
    url = f"{base}{full_path}"
    if params:
        items = [(k, v) for k, v in params.items() if v is not None]
        if items:
            url = f"{url}?{urllib.parse.urlencode(items, doseq=True)}"
    return url


def extract_markets(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the market list from a /markets response, defensively."""
    if not isinstance(payload, dict):
        return []
    markets = payload.get("markets")
    if isinstance(markets, list):
        return [m for m in markets if isinstance(m, dict)]
    return []


def filter_by_prefix(
    markets: Sequence[dict[str, Any]], prefix: str | None
) -> list[dict[str, Any]]:
    """Filter discovered markets by a ticker prefix (no-op if prefix is falsy)."""
    if not prefix:
        return list(markets)
    return [m for m in markets if str(m.get("ticker", "")).startswith(prefix)]


def apply_limit(markets: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return list(markets)
    return list(markets[:limit])


# ---------------------------------------------------------------------------
# Fetch protocol
# ---------------------------------------------------------------------------


# (status_code, parsed_json_or_text, elapsed_ms)
FetchResult = tuple[int, Any, float]
Fetch = Callable[[str, str, float], FetchResult]


def default_fetch(method: str, url: str, timeout_sec: float) -> FetchResult:
    """stdlib-only HTTP GET. Returns (status, parsed_body, elapsed_ms)."""
    _assert_public_get(method, urllib.parse.urlparse(url).path)
    req = urllib.request.Request(url, method=method)
    req.add_header("Accept", "application/json")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status = resp.getcode()
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.monotonic() - started) * 1000.0
        try:
            body: Any = json.loads(exc.read().decode("utf-8", errors="replace"))
        except Exception:
            body = None
        return exc.code, body, elapsed_ms
    elapsed_ms = (time.monotonic() - started) * 1000.0
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        body = raw
    return status, body, elapsed_ms


# ---------------------------------------------------------------------------
# Discovery + capture
# ---------------------------------------------------------------------------


def discover_markets(
    fetch: Fetch,
    *,
    base_url: str,
    status: str,
    limit: int,
    prefix: str | None,
    timeout_sec: float,
) -> tuple[list[dict[str, Any]], float]:
    """Fetch /markets and return (filtered+limited list, elapsed_ms).

    Raises ``RuntimeError`` on initial-discovery failure: a non-2xx status,
    a non-dict body, or an underlying transport error.
    """
    params: dict[str, Any] = {"status": status}
    if limit > 0:
        # request a generous server-side cap; we still apply our own limit
        # after prefix filtering.
        params["limit"] = max(limit, 100)
    url = build_url(base_url, "/markets", params)
    try:
        code, body, elapsed_ms = fetch(ALLOWED_METHOD, url, timeout_sec)
    except urllib.error.URLError as exc:  # transport / DNS / timeout
        raise RuntimeError(f"markets discovery transport error: {exc}") from exc
    if code < 200 or code >= 300:
        raise RuntimeError(f"markets discovery failed: status={code}")
    if not isinstance(body, dict):
        raise RuntimeError("markets discovery returned non-dict payload")
    markets = extract_markets(body)
    markets = filter_by_prefix(markets, prefix)
    markets = apply_limit(markets, limit)
    return markets, elapsed_ms


def fetch_market(
    fetch: Fetch, *, base_url: str, ticker: str, timeout_sec: float
) -> tuple[dict[str, Any] | None, float, str | None]:
    """Fetch /markets/{ticker}. Returns (market_dict_or_none, elapsed_ms, error)."""
    url = build_url(base_url, f"/markets/{ticker}")
    try:
        code, body, elapsed_ms = fetch(ALLOWED_METHOD, url, timeout_sec)
    except Exception as exc:
        return None, 0.0, f"transport: {exc!r}"
    if code < 200 or code >= 300:
        return None, elapsed_ms, f"status={code}"
    if not isinstance(body, dict):
        return None, elapsed_ms, "non-dict body"
    market = body.get("market") if isinstance(body.get("market"), dict) else body
    return market, elapsed_ms, None


def fetch_orderbook(
    fetch: Fetch, *, base_url: str, ticker: str, timeout_sec: float
) -> tuple[dict[str, Any] | None, float, str | None]:
    """Fetch /markets/{ticker}/orderbook. Returns (orderbook_or_none, elapsed_ms, error)."""
    url = build_url(base_url, f"/markets/{ticker}/orderbook")
    try:
        code, body, elapsed_ms = fetch(ALLOWED_METHOD, url, timeout_sec)
    except Exception as exc:
        return None, 0.0, f"transport: {exc!r}"
    if code < 200 or code >= 300:
        return None, elapsed_ms, f"status={code}"
    if not isinstance(body, dict):
        return None, elapsed_ms, "non-dict body"
    return body, elapsed_ms, None


# ---------------------------------------------------------------------------
# Record building + write
# ---------------------------------------------------------------------------


def build_record(
    *,
    base_url: str,
    status_filter: str,
    ticker: str,
    market: dict[str, Any] | None,
    orderbook: dict[str, Any] | None,
    markets_fetch_ms: float | None,
    market_fetch_ms: float | None,
    orderbook_fetch_ms: float | None,
    error: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source": SOURCE_TAG,
        "captured_wall_ts_ns": time.time_ns(),
        "captured_monotonic_ns": time.monotonic_ns(),
        "base_url": base_url,
        "status_filter": status_filter,
        "ticker": ticker,
    }
    if market is not None:
        record["market"] = market
    if orderbook is not None:
        record["orderbook"] = orderbook
    if markets_fetch_ms is not None:
        record["markets_fetch_ms"] = round(markets_fetch_ms, 3)
    if market_fetch_ms is not None:
        record["market_fetch_ms"] = round(market_fetch_ms, 3)
    if orderbook_fetch_ms is not None:
        record["orderbook_fetch_ms"] = round(orderbook_fetch_ms, 3)
    if error is not None:
        record["error"] = error
    return record


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Append JSONL records to ``path``, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_output_path(out_dir: Path | None = None) -> Path:
    base = out_dir or DEFAULT_OUTPUT_DIR
    today = time.strftime("%Y-%m-%d", time.gmtime())
    return base / f"{today}.jsonl"


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kalshi_rest_snapshot",
        description="Public Kalshi REST snapshot tool (read-only, no auth).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("KALSHI_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--status", default="open")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--series-prefix", default=None)
    parser.add_argument(
        "--ticker",
        action="append",
        default=None,
        help="Repeatable. If given, skip discovery and capture only these tickers.",
    )
    parser.add_argument("--sleep-sec", type=float, default=0.25)
    parser.add_argument("--out", default=None, help="Output JSONL path.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-orderbook",
        dest="include_orderbook",
        action="store_false",
        default=True,
    )
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    return parser.parse_args(argv)


def run(
    argv: Sequence[str] | None = None,
    *,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    do_fetch = fetch or default_fetch

    out_path = Path(args.out) if args.out else default_output_path()

    explicit_tickers: list[str] = list(args.ticker) if args.ticker else []
    records: list[dict[str, Any]] = []

    if explicit_tickers:
        markets_fetch_ms: float | None = None
        target_tickers = explicit_tickers
    else:
        try:
            markets, markets_fetch_ms = discover_markets(
                do_fetch,
                base_url=args.base_url,
                status=args.status,
                limit=args.limit,
                prefix=args.series_prefix,
                timeout_sec=args.timeout_sec,
            )
        except RuntimeError as exc:
            print(f"discovery failed: {exc}", file=sys.stderr)
            return 2
        target_tickers = [str(m.get("ticker", "")) for m in markets if m.get("ticker")]

    summary = {
        "tickers_to_capture": len(target_tickers),
        "include_orderbook": args.include_orderbook,
        "out": str(out_path),
        "dry_run": args.dry_run,
    }

    for idx, ticker in enumerate(target_tickers):
        if explicit_tickers:
            market, market_ms, m_err = fetch_market(
                do_fetch,
                base_url=args.base_url,
                ticker=ticker,
                timeout_sec=args.timeout_sec,
            )
        else:
            # discovery already gave us the market dict; avoid re-fetching
            market = next(
                (m for m in markets if str(m.get("ticker")) == ticker), None
            )
            market_ms = None
            m_err = None

        ob: dict[str, Any] | None = None
        ob_ms: float | None = None
        ob_err: str | None = None
        if args.include_orderbook and m_err is None:
            ob, ob_ms, ob_err = fetch_orderbook(
                do_fetch,
                base_url=args.base_url,
                ticker=ticker,
                timeout_sec=args.timeout_sec,
            )

        error_field = m_err if m_err else ob_err

        records.append(
            build_record(
                base_url=args.base_url,
                status_filter=args.status,
                ticker=ticker,
                market=market,
                orderbook=ob,
                markets_fetch_ms=markets_fetch_ms if idx == 0 else None,
                market_fetch_ms=market_ms,
                orderbook_fetch_ms=ob_ms,
                error=error_field,
            )
        )

        if idx + 1 < len(target_tickers) and args.sleep_sec > 0:
            sleep(args.sleep_sec)

    if args.dry_run:
        summary["records_built"] = len(records)
        print(json.dumps(summary, sort_keys=True))
        return 0

    written = append_jsonl(out_path, records)
    summary["records_written"] = written
    print(json.dumps(summary, sort_keys=True))
    return 0


def main() -> None:  # pragma: no cover - thin shim
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
