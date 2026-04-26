"""Phase 5a.1.1 — Kalshi WebSocket public-data recorder.

Manual, single-shot, duration-bounded recorder that subscribes ONLY to the
public read-only channels ``trade`` and ``orderbook_delta`` and writes
append-only JSONL to disk for passive analysis.

Hard safety guards (see ``ALLOWED_WS_CHANNELS``, ``_assert_public_channel``,
``_guarded_subscribe``, and ``_orderbook_payload_anomalies``):

- Channel allowlist is a frozenset of public read-only channels:
  ``{"trade", "orderbook_delta"}``. Every other Kalshi WS channel is
  rejected at argparse-driven validation BEFORE auth material is loaded,
  AND again at every subscribe call site through ``_guarded_subscribe``.
  Defense in depth means an injected/fake WebSocket cannot bypass the
  guard.
- ``orderbook_delta``-channel frames pass through a per-frame payload
  guard that recursively walks key paths (never values) and rejects
  any frame whose top-level keys, ``msg`` keys, or any nested key path
  matches a forbidden substring set (``order``, ``account``, ``user``,
  ``balance``, ``position``, ``fill``, ``portfolio``, ``owner``,
  ``holder``, ``customer``, ``client``). Anomalous frames are
  quarantined: not written to JSONL, not written to any other file,
  ``anomalous_count`` is incremented, ``frame_seq`` is NOT advanced,
  and a safe key-path-only summary is logged to stderr (rate-limited
  to the first ANOMALY_LOG_LIMIT anomalies). Trade-channel frames
  bypass this guard.
- Recorder reads ONLY the recorder-specific environment variables
  ``KALSHI_RECORDER_API_KEY_ID`` and ``KALSHI_RECORDER_PRIVATE_KEY_PATH``.
  It does NOT fall back to ``KALSHI_API_KEY_ID``/``KALSHI_PRIVATE_KEY_PATH``,
  it does NOT use the venue adapter's env-loading helper, and it does NOT
  parse ``/root/executor/.env``.
- Non-dry-run execution requires ``--i-confirm-read-only-key``. This is an
  operator self-attestation that the configured recorder API key has
  read-only scope; the recorder cannot programmatically verify scope and
  prints a stderr warning before opening the WebSocket.

Module-level imports never load ``executor.venue_adapters.kalshi.adapter``
or ``...rest`` — ``KalshiWS`` and ``KalshiAuth`` are imported lazily inside
``_default_ws_factory`` so test imports of this module stay narrow.

Run as a module::

    python3 -m executor.tools.kalshi_ws_snapshot \\
        --ticker KXFED-26-NOV \\
        --max-duration-sec 60 \\
        --max-messages 5000 \\
        --i-confirm-read-only-key

Dry-run path validates args/policy without opening a WebSocket and without
writing output::

    python3 -m executor.tools.kalshi_ws_snapshot --ticker FOO --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Channel policy
# ---------------------------------------------------------------------------

ALLOWED_WS_CHANNELS: frozenset[str] = frozenset({"trade", "orderbook_delta"})

# Channels we explicitly call out as forbidden in error messages so an
# operator running into the guard can see what they tried to subscribe to.
# This list is illustrative; the actual rejection rule is "anything not in
# ALLOWED_WS_CHANNELS", so unknown channels are rejected too.
FORBIDDEN_WS_CHANNELS: frozenset[str] = frozenset({
    "fill",
    "position",
    "position_update",
    "market_positions",
    "communications",
    "order_group_updates",
    "orders",
    "balance",
    "portfolio",
})


# ---------------------------------------------------------------------------
# Orderbook payload guard policy
# ---------------------------------------------------------------------------

# Top-level keys allowed on an orderbook_delta-channel frame. Empirically
# validated against the recorder-key live capture in phase 5a.1.1 diagnostic.
ALLOWED_TOP_LEVEL_ORDERBOOK_KEYS: frozenset[str] = frozenset({
    "type", "sid", "seq", "msg",
})

# Superset of keys allowed under msg for orderbook_snapshot + orderbook_delta
# frames. Snapshot-only fields (yes_dollars_fp / no_dollars_fp) and delta-only
# fields (price_dollars / delta_fp / side / ts / ts_ms) are both allowed; the
# guard validates that observed keys are a SUBSET of this superset, not that
# all keys are present.
ALLOWED_ORDERBOOK_MSG_KEYS: frozenset[str] = frozenset({
    "market_ticker",
    "market_id",
    "yes_dollars_fp",
    "no_dollars_fp",
    "price_dollars",
    "delta_fp",
    "side",
    "ts",
    "ts_ms",
})

# Forbidden substrings checked against KEY PATHS (key names only, never
# values). The value "orderbook_delta" appearing in a `type` field will not
# trip the guard because the walker yields key paths only.
FORBIDDEN_KEY_PATH_SUBSTRINGS: tuple[str, ...] = (
    "order",
    "account",
    "user",
    "balance",
    "position",
    "fill",
    "portfolio",
    "owner",
    "holder",
    "customer",
    "client",
)

# Cap on detailed anomaly stderr logs per recorder run; counter still
# increments past this point but suppressed log count is reported in the
# final summary.
ANOMALY_LOG_LIMIT: int = 10


# ---------------------------------------------------------------------------
# Output and credential constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("audit-logs/market_snapshots/kalshi_ws")
SOURCE_TAG = "kalshi_ws_readonly"

RECORDER_API_KEY_ENV = "KALSHI_RECORDER_API_KEY_ID"
RECORDER_PRIVATE_KEY_PATH_ENV = "KALSHI_RECORDER_PRIVATE_KEY_PATH"

READ_ONLY_FLAG = "--i-confirm-read-only-key"
READ_ONLY_WARNING = (
    "WARNING: Operator has confirmed via --i-confirm-read-only-key that the "
    "recorder API key (KALSHI_RECORDER_API_KEY_ID) has read-only scope. If "
    "this key has write scope, accidental order placement is possible despite "
    "the channel allowlist. Recorder will proceed."
)


# ---------------------------------------------------------------------------
# Channel guard chokepoints
# ---------------------------------------------------------------------------


def _assert_public_channel(channel: str) -> None:
    """Reject any channel not in the trade-only allowlist.

    Single chokepoint between argparse-driven validation and every WebSocket
    subscribe call. Both ``run`` (argparse stage) and ``_guarded_subscribe``
    invoke it; the latter ensures even a fake/injected WS cannot bypass the
    guard during a test.
    """
    if channel not in ALLOWED_WS_CHANNELS:
        raise ValueError(
            f"forbidden channel {channel!r}: kalshi_ws_snapshot only permits "
            f"channels in {sorted(ALLOWED_WS_CHANNELS)}; "
            f"explicitly rejected examples: {sorted(FORBIDDEN_WS_CHANNELS)}"
        )


async def _guarded_subscribe(
    ws: Any,
    channels: Sequence[str],
    tickers: Sequence[str],
) -> Any:
    """Apply ``_assert_public_channel`` to every channel BEFORE delegating to
    ``ws.subscribe``.

    All call sites that subscribe (live or fake WS) go through this helper.
    Even a test fake that records subscribe calls without enforcement cannot
    receive a forbidden channel because the wrapper raises first.
    """
    for ch in channels:
        _assert_public_channel(ch)
    return await ws.subscribe(channels, list(tickers))


# ---------------------------------------------------------------------------
# Default WS factory (lazy imports — module import never loads adapter/rest)
# ---------------------------------------------------------------------------


def _default_ws_factory(api_key_id: str, private_key_path: str) -> Any:
    """Construct a real KalshiWS bound to a recorder-credential KalshiAuth.

    Imports of ``KalshiAuth``/``KalshiWS`` are deferred to call time so that
    importing ``executor.tools.kalshi_ws_snapshot`` does NOT pull in
    ``executor.venue_adapters.kalshi.adapter`` or ``...rest`` (the kalshi
    package ``__init__`` re-exports the adapter).
    """
    # Local imports — see module docstring.
    from executor.venue_adapters.kalshi.auth import KalshiAuth
    from executor.venue_adapters.kalshi.websocket import KalshiWS

    auth = KalshiAuth(api_key_id=api_key_id, private_key_path=private_key_path)
    return KalshiWS(auth=auth)


# ---------------------------------------------------------------------------
# Recursive key-path walker + orderbook payload guard
# ---------------------------------------------------------------------------


def _walk_key_paths(obj: Any, prefix: str = "") -> list[str]:
    """Recursively yield all dict key paths in ``obj`` as dotted strings.

    Returns a sorted, deduplicated ``list[str]``. Strings only — values are
    never yielded or inspected. Handles arbitrarily nested dicts, lists, and
    dicts inside lists. Non-string dict keys are skipped (the guard does not
    claim to validate non-string keys; downstream allowlist still rejects
    any unknown-shaped frame).
    """
    out: set[str] = set()

    def _rec(o: Any, p: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if not isinstance(k, str):
                    continue
                path = f"{p}.{k}" if p else k
                out.add(path)
                _rec(v, path)
        elif isinstance(o, list):
            for item in o:
                _rec(item, p)

    _rec(obj, prefix)
    return sorted(out)


def _orderbook_payload_anomalies(frame: dict[str, Any]) -> list[str]:
    """Validate an orderbook_delta-channel frame's keys against policy.

    Returns a sorted list of rejected key paths. Empty list means the frame
    is clean. Rejection reasons (combined into one set):

    - Top-level key not in ``ALLOWED_TOP_LEVEL_ORDERBOOK_KEYS``.
    - Key directly under ``msg`` (when ``msg`` is a dict) not in
      ``ALLOWED_ORDERBOOK_MSG_KEYS``.
    - Any nested key path containing a substring in
      ``FORBIDDEN_KEY_PATH_SUBSTRINGS`` (case-insensitive, key names only).

    The guard inspects key NAMES, not values. A frame whose ``type`` value
    is the string ``"orderbook_delta"`` will not trip the guard, because
    the walker never yields values.

    Snapshot frames legitimately omit delta-only fields, and delta frames
    legitimately omit snapshot-only fields; this is a SUPERSET allowlist
    check, not a required-field check.
    """
    rejected: set[str] = set()

    # Structural: top-level keys must be a subset of the allowed set.
    for k in frame.keys():
        if not isinstance(k, str) or k not in ALLOWED_TOP_LEVEL_ORDERBOOK_KEYS:
            rejected.add(str(k))

    # Structural: msg-level keys (only when msg is a dict). If msg is not a
    # dict we don't add a rejection here — the top-level allowlist already
    # accepts msg, and the walker below won't add msg.* paths for non-dict
    # bodies. A scalar msg is unusual but not by itself a guard violation.
    body = frame.get("msg")
    if isinstance(body, dict):
        for k in body.keys():
            if not isinstance(k, str) or k not in ALLOWED_ORDERBOOK_MSG_KEYS:
                rejected.add(f"msg.{k!s}")

    # Substring scan recursively across every key path. Catches deeply
    # nested forbidden keys (e.g. msg.foo.account_id) regardless of whether
    # the structural allowlist would also reject the parent.
    for path in _walk_key_paths(frame):
        path_lc = path.lower()
        if any(sub in path_lc for sub in FORBIDDEN_KEY_PATH_SUBSTRINGS):
            rejected.add(path)

    return sorted(rejected)


# ---------------------------------------------------------------------------
# Frame -> record
# ---------------------------------------------------------------------------


def _channel_for_msg_type(mtype: str, channels: Sequence[str]) -> str:
    """Map a Kalshi message type to the subscription channel that owns it.

    - ``"trade"`` belongs to the trade channel.
    - ``"orderbook_snapshot"`` and ``"orderbook_delta"`` both belong to the
      orderbook_delta channel (the Kalshi WS dispatches both message types
      through a single ``orderbook_delta`` subscription per
      ``executor.venue_adapters.kalshi.websocket._channel_matches``).
    - Error / unknown frames (``"error"``, malformed types) fall back to
      the first subscribed channel. Error frames are fanned out to every
      subscription in the underlying WS client, so labeling them with the
      recorder's first subscribed channel is a defensible pin.
    """
    if mtype == "trade" and "trade" in channels:
        return "trade"
    if mtype in ("orderbook_snapshot", "orderbook_delta") and "orderbook_delta" in channels:
        return "orderbook_delta"
    return channels[0] if channels else "trade"


def _frame_to_record(
    msg: Any,
    *,
    frame_seq: int,
    session_id: str,
    clock_wall_ns: Callable[[], int],
    clock_monotonic_ns: Callable[[], int],
    channels: Sequence[str],
) -> tuple[dict[str, Any] | None, bool, bool]:
    """Convert a queue frame to a JSONL record.

    Returns ``(record_or_None, is_malformed, is_error)``.

    - Non-dict, non-Exception frames (strings, ints, lists, ...) are
      malformed: record is None, ``is_malformed=True``.
    - Exception frames (e.g. ``VenueError`` mapped from a Kalshi error frame)
      are recorded with ``error=true``.
    - Dict frames with ``type=="error"`` are recorded with ``error=true`` and
      the raw dict.
    - Dict frames with any other type are recorded as normal data records.

    The ``channel`` field is derived from ``type`` via
    ``_channel_for_msg_type`` so that orderbook_snapshot and orderbook_delta
    frames are both labeled ``channel="orderbook_delta"``.
    """
    if isinstance(msg, BaseException):
        return ({
            "source": SOURCE_TAG,
            "channel": _channel_for_msg_type("error", channels),
            "msg_type": "error",
            "market_ticker": None,
            "captured_wall_ts_ns": clock_wall_ns(),
            "received_monotonic_ns": clock_monotonic_ns(),
            "session_id": session_id,
            "frame_seq": frame_seq,
            "kalshi_seq": None,
            "raw": repr(msg),
            "error": True,
        }, False, True)
    if not isinstance(msg, dict):
        return (None, True, False)
    mtype = msg.get("type")
    if mtype == "error":
        return ({
            "source": SOURCE_TAG,
            "channel": _channel_for_msg_type("error", channels),
            "msg_type": "error",
            "market_ticker": None,
            "captured_wall_ts_ns": clock_wall_ns(),
            "received_monotonic_ns": clock_monotonic_ns(),
            "session_id": session_id,
            "frame_seq": frame_seq,
            "kalshi_seq": None,
            "raw": msg,
            "error": True,
        }, False, True)
    body = msg.get("msg") if isinstance(msg.get("msg"), dict) else None
    ticker = body.get("market_ticker") if isinstance(body, dict) else None
    seq_val = msg.get("seq")
    kalshi_seq: int | None = seq_val if isinstance(seq_val, int) else None
    mtype_str = str(mtype) if mtype is not None else ""
    return ({
        "source": SOURCE_TAG,
        "channel": _channel_for_msg_type(mtype_str, channels),
        "msg_type": mtype_str,
        "market_ticker": ticker,
        "captured_wall_ts_ns": clock_wall_ns(),
        "received_monotonic_ns": clock_monotonic_ns(),
        "session_id": session_id,
        "frame_seq": frame_seq,
        "kalshi_seq": kalshi_seq,
        "raw": msg,
    }, False, False)


# ---------------------------------------------------------------------------
# Drive loop
# ---------------------------------------------------------------------------


async def _drive(
    *,
    ws_factory: Callable[[str, str], Any],
    api_key_id: str,
    private_key_path: str,
    channels: Sequence[str],
    tickers: Sequence[str],
    max_duration_sec: float,
    max_messages: int,
    out_path: Path,
    session_id: str,
    clock_wall_ns: Callable[[], int],
    clock_monotonic_ns: Callable[[], int],
    async_sleep: Callable[[float], Awaitable[None]],
) -> tuple[int, int, int, int]:
    """Subscribe via guarded chokepoint, drain the queue, append JSONL.

    Single-shot: returns when ``max_messages`` or ``max_duration_sec`` is
    reached, or when the subscription queue closes (sentinel ``None``).

    Returns ``(records_written, malformed_count, anomalous_count,
    suppressed_log_count)``. Anomalous orderbook_delta frames are
    quarantined: not written to the JSONL, not written to any other file,
    and ``frame_seq`` is NOT incremented for them, matching the existing
    malformed-frame policy.
    """
    ws = ws_factory(api_key_id, private_key_path)
    await ws.start()
    try:
        sub = await _guarded_subscribe(ws, channels, tickers)
        deadline_mono_ns = (
            clock_monotonic_ns() + int(max_duration_sec * 1_000_000_000)
        )
        frame_seq = 0
        malformed = 0
        anomalous = 0
        suppressed = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            while frame_seq < max_messages:
                if clock_monotonic_ns() >= deadline_mono_ns:
                    break
                try:
                    msg = sub.queue.get_nowait()
                except asyncio.QueueEmpty:
                    await async_sleep(0.05)
                    continue
                if msg is None:
                    break
                rec, is_malformed, _is_error = _frame_to_record(
                    msg,
                    frame_seq=frame_seq,
                    session_id=session_id,
                    clock_wall_ns=clock_wall_ns,
                    clock_monotonic_ns=clock_monotonic_ns,
                    channels=channels,
                )
                if is_malformed:
                    malformed += 1
                    continue
                assert rec is not None  # mypy: only None when malformed
                # Orderbook payload guard: applies ONLY to orderbook_delta
                # channel data frames. Trade frames bypass entirely. Error
                # frames bypass too — they have no Kalshi msg/body to scan
                # and are wrapped through _frame_to_record's error path.
                if (
                    rec["channel"] == "orderbook_delta"
                    and not rec.get("error")
                    and isinstance(msg, dict)
                ):
                    anomalies = _orderbook_payload_anomalies(msg)
                    if anomalies:
                        anomalous += 1
                        if anomalous <= ANOMALY_LOG_LIMIT:
                            print(
                                json.dumps(
                                    {
                                        "event": "kalshi_ws_recorder.orderbook_anomaly",
                                        "anomalous_count": anomalous,
                                        "rejected_key_paths": anomalies,
                                    },
                                    sort_keys=True,
                                ),
                                file=sys.stderr,
                            )
                        else:
                            suppressed += 1
                        # Quarantine: do NOT write anywhere, do NOT
                        # increment frame_seq, continue capture.
                        continue
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                fh.flush()
                frame_seq += 1
        return frame_seq, malformed, anomalous, suppressed
    finally:
        await ws.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_output_path(out_dir: Path | None = None) -> Path:
    base = out_dir or DEFAULT_OUTPUT_DIR
    today = time.strftime("%Y-%m-%d", time.gmtime())
    return base / f"{today}.jsonl"


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kalshi_ws_snapshot",
        description=(
            "Manual, single-shot Kalshi WebSocket public-data recorder. "
            "Read-only — only the public 'trade' and 'orderbook_delta' "
            "channels are permitted; orderbook_delta frames pass through a "
            "key-path payload guard that quarantines anomalies. Auth uses "
            "recorder-specific environment variables only."
        ),
    )
    parser.add_argument(
        "--ticker",
        action="append",
        default=None,
        required=False,
        help="Repeatable. Required at least once — no firehose subscription.",
    )
    parser.add_argument(
        "--channel",
        action="append",
        default=None,
        help=(
            "Repeatable. Default 'trade'. Only 'trade' and 'orderbook_delta' "
            "are accepted; any other channel (fill, position, balance, "
            "portfolio, ...) is rejected."
        ),
    )
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=60.0,
        help="Stop after this many wall seconds. Default 60.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=5000,
        help="Stop after this many recorded messages. Default 5000.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output JSONL path. Default: "
            "audit-logs/market_snapshots/kalshi_ws/YYYY-MM-DD.jsonl"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate args and channel policy only. Does NOT open the "
            "WebSocket and does NOT write output. Credentials are not read."
        ),
    )
    parser.add_argument(
        "--i-confirm-read-only-key",
        dest="i_confirm_read_only_key",
        action="store_true",
        help=(
            "Required for non-dry-run. Operator self-attestation that the "
            "recorder API key has read-only scope. Recorder cannot verify "
            "scope programmatically."
        ),
    )
    return parser.parse_args(list(argv))


def _resolve_out_path(out_arg: str | None) -> Path:
    return Path(out_arg) if out_arg else default_output_path()


def run(
    argv: Sequence[str] | None = None,
    *,
    ws_factory: Callable[[str, str], Any] | None = None,
    env: Mapping[str, str] | None = None,
    clock_wall_ns: Callable[[], int] | None = None,
    clock_monotonic_ns: Callable[[], int] | None = None,
    async_sleep: Callable[[float], Awaitable[None]] | None = None,
    session_id: str | None = None,
) -> int:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])

    # --- Step 1: argparse validation. Channel/ticker checks BEFORE auth.
    channels: list[str] = list(args.channel) if args.channel else ["trade"]
    for ch in channels:
        # Forbidden-channel rejection happens here, before any credential
        # read and before WebSocket creation. This is the first chokepoint.
        _assert_public_channel(ch)

    if not args.ticker:
        print(
            "--ticker is required (at least one). Refusing to subscribe "
            "without an explicit ticker filter — no firehose.",
            file=sys.stderr,
        )
        return 2
    tickers: list[str] = list(args.ticker)

    out_path = _resolve_out_path(args.out)

    # --- Step 2: dry-run short-circuit. No env, no WS, no output.
    if args.dry_run:
        summary = {
            "mode": "dry-run",
            "channels": channels,
            "tickers": tickers,
            "max_duration_sec": args.max_duration_sec,
            "max_messages": args.max_messages,
            "out": str(out_path),
        }
        print(json.dumps(summary, sort_keys=True))
        return 0

    # --- Step 3: read-only acknowledgement gate. Must come BEFORE auth load.
    if not args.i_confirm_read_only_key:
        print(
            f"refusing to run without {READ_ONLY_FLAG}. "
            "Recorder requires explicit operator confirmation that the "
            f"configured API key ({RECORDER_API_KEY_ENV}) has read-only "
            "scope. Pass --dry-run to validate args without WebSocket I/O.",
            file=sys.stderr,
        )
        return 2

    # Print warning regardless of whether scope can be programmatically
    # verified. Operator self-attestation is the primary trust mechanism.
    print(READ_ONLY_WARNING, file=sys.stderr)

    # --- Step 4: load recorder-specific credentials (only after channel
    # validation and the read-only ack). Generic Kalshi env names are NOT
    # read.
    env_map: Mapping[str, str] = env if env is not None else os.environ
    api_key_id = env_map.get(RECORDER_API_KEY_ENV)
    private_key_path = env_map.get(RECORDER_PRIVATE_KEY_PATH_ENV)
    if not api_key_id:
        print(
            f"missing {RECORDER_API_KEY_ENV} in environment — recorder "
            "uses recorder-specific credentials only and does not fall "
            "back to KALSHI_API_KEY_ID.",
            file=sys.stderr,
        )
        return 2
    if not private_key_path:
        print(
            f"missing {RECORDER_PRIVATE_KEY_PATH_ENV} in environment — "
            "recorder uses recorder-specific credentials only and does not "
            "fall back to KALSHI_PRIVATE_KEY_PATH.",
            file=sys.stderr,
        )
        return 2

    factory = ws_factory if ws_factory is not None else _default_ws_factory
    sid = session_id or str(uuid.uuid4())
    wall_ns = clock_wall_ns or time.time_ns
    mono_ns = clock_monotonic_ns or time.monotonic_ns
    sleep_fn = async_sleep or asyncio.sleep

    written, malformed, anomalous, suppressed = asyncio.run(_drive(
        ws_factory=factory,
        api_key_id=api_key_id,
        private_key_path=private_key_path,
        channels=channels,
        tickers=tickers,
        max_duration_sec=args.max_duration_sec,
        max_messages=args.max_messages,
        out_path=out_path,
        session_id=sid,
        clock_wall_ns=wall_ns,
        clock_monotonic_ns=mono_ns,
        async_sleep=sleep_fn,
    ))

    summary: dict[str, Any] = {
        "mode": "live",
        "session_id": sid,
        "channels": channels,
        "tickers": tickers,
        "out": str(out_path),
        "records_written": written,
        "malformed_count": malformed,
        "anomalous_count": anomalous,
    }
    if suppressed > 0:
        summary["anomalies_suppressed_log_count"] = suppressed
    print(json.dumps(summary, sort_keys=True))
    return 0


def main() -> None:  # pragma: no cover - thin shim
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
