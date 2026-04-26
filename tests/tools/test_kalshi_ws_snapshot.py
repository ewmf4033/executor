"""Tests for executor.tools.kalshi_ws_snapshot.

Phase 5a.1.0 — narrow scope. Trade-only channel allowlist; recorder-specific
credential names; non-dry-run requires --i-confirm-read-only-key.

No live network. The drive loop and ``_default_ws_factory`` are bypassed by
injecting a ``FakeWS``. Every test is hermetic.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from executor.tools import kalshi_ws_snapshot as wsmod


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class FakeQueue:
    """Sync drop-in for asyncio.Queue.get_nowait().

    Backed by a list. Raises ``asyncio.QueueEmpty`` when drained, matching
    the real queue's exception so the recorder's polling fallback fires.
    """

    def __init__(self, items: list[Any] | None = None) -> None:
        self._items = list(items or [])

    def put(self, item: Any) -> None:
        self._items.append(item)

    def get_nowait(self) -> Any:
        if not self._items:
            raise asyncio.QueueEmpty()
        return self._items.pop(0)


class FakeSub:
    def __init__(self, queue: FakeQueue) -> None:
        self.queue = queue


class FakeWS:
    """Records every interaction; never opens a real socket."""

    def __init__(self, frames: list[Any] | None = None) -> None:
        self._queue = FakeQueue(frames)
        self.start_called = False
        self.stop_called = False
        self.subscribe_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    async def start(self) -> None:
        self.start_called = True

    async def stop(self) -> None:
        self.stop_called = True

    async def subscribe(self, channels: Any, market_tickers: Any) -> FakeSub:
        self.subscribe_calls.append((tuple(channels), tuple(market_tickers)))
        return FakeSub(self._queue)


class RecordingEnv:
    """Mapping that records every ``.get()`` key the recorder queries."""

    def __init__(self, base: dict[str, str] | None = None) -> None:
        self._base = dict(base or {})
        self.gets: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:
        self.gets.append(key)
        return self._base.get(key, default)

    def __getitem__(self, key: str) -> str:
        self.gets.append(key)
        return self._base[key]


class MockClock:
    """Monotonic clock whose value is advanced explicitly by tests."""

    def __init__(self, start_ns: int = 0) -> None:
        self.now_ns = start_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance_sec(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1_000_000_000)


def _make_advancing_sleep(clock: MockClock):
    async def _sleep(seconds: float) -> None:
        clock.advance_sec(seconds)
    return _sleep


def _trade_frame(ticker: str, *, seq: int = 0, price: int = 50) -> dict[str, Any]:
    return {
        "type": "trade",
        "seq": seq,
        "msg": {
            "market_ticker": ticker,
            "yes_price": price,
            "count": 1,
            "side": "yes",
        },
    }


# ---------------------------------------------------------------------------
# Credential / import posture
# ---------------------------------------------------------------------------


# 1. Importing module does not read env eagerly.
def test_importing_module_does_not_read_env_eagerly() -> None:
    """Module import must not call os.environ — verified in a clean subprocess.

    The subprocess installs an audit hook that records every os.environ
    access. A bare ``import`` of the recorder must not trigger any.
    """
    code = textwrap.dedent(
        """
        import sys, os
        accesses = []
        # Replace os.environ with a tracking dict before importing the module.
        class _TrackingEnv(dict):
            def __getitem__(self, k):
                accesses.append(("getitem", k))
                return super().__getitem__(k)
            def get(self, k, default=None):
                accesses.append(("get", k))
                return super().get(k, default)
        original = dict(os.environ)
        os.environ = _TrackingEnv(original)  # type: ignore[assignment]
        sys.path.insert(0, %(repo)r)
        import executor.tools.kalshi_ws_snapshot  # noqa: F401
        # Print the accesses; an empty list means no eager env reads.
        print(repr(accesses))
        """
    ) % {"repo": str(REPO_ROOT)}
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "[]", f"unexpected eager env reads at import: {out}"


# 2. Tool reads only KALSHI_RECORDER_API_KEY_ID and KALSHI_RECORDER_PRIVATE_KEY_PATH.
def test_only_recorder_specific_env_vars_are_read(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "rec-id",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
        # Generic Kalshi vars are present — recorder MUST NOT read them.
        "KALSHI_API_KEY_ID": "generic-id",
        "KALSHI_PRIVATE_KEY_PATH": "/etc/kalshi.key",
    })
    fake_ws = FakeWS(frames=[None])  # sentinel closes the loop quickly
    clock = MockClock()
    rc = wsmod.run(
        [
            "--ticker", "T",
            "--max-duration-sec", "5",
            "--max-messages", "10",
            "--out", str(out),
            "--i-confirm-read-only-key",
        ],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
        clock_wall_ns=lambda: 1_000_000_000,
        clock_monotonic_ns=clock,
        async_sleep=_make_advancing_sleep(clock),
    )
    assert rc == 0
    # The recorder may query the recorder-specific names; it must NEVER query
    # the generic Kalshi names.
    assert "KALSHI_API_KEY_ID" not in env.gets
    assert "KALSHI_PRIVATE_KEY_PATH" not in env.gets
    assert set(env.gets) <= {
        "KALSHI_RECORDER_API_KEY_ID",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH",
    }


# 3. Tool refuses/fails if only generic creds are set.
def test_refuses_when_only_generic_kalshi_creds_set(tmp_path: Path, capsys) -> None:
    out = tmp_path / "out.jsonl"
    env = RecordingEnv({
        "KALSHI_API_KEY_ID": "generic-id",
        "KALSHI_PRIVATE_KEY_PATH": "/etc/kalshi.key",
        # Recorder-specific names absent.
    })
    fake_ws = FakeWS(frames=[])
    rc = wsmod.run(
        [
            "--ticker", "T",
            "--out", str(out),
            "--i-confirm-read-only-key",
        ],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
    )
    assert rc != 0
    assert not out.exists()
    # WS must not have been started/subscribed.
    assert fake_ws.start_called is False
    assert fake_ws.subscribe_calls == []
    err = capsys.readouterr().err
    assert "KALSHI_RECORDER_API_KEY_ID" in err


# 4. Tool does not call auth_from_env().
def test_source_does_not_reference_auth_from_env() -> None:
    """Static check: the recorder source must not import or call auth_from_env."""
    src = Path(wsmod.__file__).read_text(encoding="utf-8")
    # Strip the module docstring so a doc-only mention wouldn't count, though
    # the recorder docstring also avoids the bare symbol.
    assert "auth_from_env" not in src, (
        "recorder must not reference auth_from_env() — recorder constructs "
        "KalshiAuth directly from recorder-specific env values"
    )


# 5. Importing module does not load executor.venue_adapters.kalshi.rest or .adapter.
def test_module_import_does_not_load_kalshi_adapter_or_rest() -> None:
    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %(repo)r)
        import executor.tools.kalshi_ws_snapshot  # noqa: F401
        loaded = sorted(m for m in sys.modules if m.startswith("executor.venue_adapters.kalshi"))
        print("\\n".join(loaded))
        """
    ) % {"repo": str(REPO_ROOT)}
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    loaded = set(out.splitlines()) if out else set()
    forbidden = {
        "executor.venue_adapters.kalshi.adapter",
        "executor.venue_adapters.kalshi.rest",
    }
    assert not (loaded & forbidden), (
        f"recorder import must not load adapter/rest; loaded={sorted(loaded)}"
    )


# 6. No order/cancel/amend/replace symbols reachable from tool module.
def test_no_order_mutation_symbols_in_tool_module() -> None:
    forbidden_names = (
        "place_order",
        "cancel_order",
        "amend_order",
        "replace_order",
        "submit_order",
    )
    public_names = [n for n in dir(wsmod) if not n.startswith("_")]
    for name in forbidden_names:
        assert name not in public_names, (
            f"recorder must not expose {name!r}; saw it on the module"
        )
    src = Path(wsmod.__file__).read_text(encoding="utf-8")
    for token in ("place_order", "cancel_order", "amend_order", "replace_order"):
        assert token not in src, (
            f"recorder source must not reference {token!r}"
        )


# ---------------------------------------------------------------------------
# Channel safety
# ---------------------------------------------------------------------------


# 7. Only trade accepted.
def test_assert_public_channel_accepts_trade() -> None:
    wsmod._assert_public_channel("trade")


# 8. orderbook_delta rejected.
def test_assert_public_channel_rejects_orderbook_delta() -> None:
    with pytest.raises(ValueError, match="forbidden channel"):
        wsmod._assert_public_channel("orderbook_delta")


# 9. fill/position/.../unknown rejected.
@pytest.mark.parametrize(
    "channel",
    [
        "fill",
        "position",
        "position_update",
        "market_positions",
        "communications",
        "order_group_updates",
        "orders",
        "balance",
        "portfolio",
        "lifecycle",          # unknown
        "settled_market_market",  # unknown
        "",                   # empty
        "TRADE",              # case-sensitive — must reject
    ],
)
def test_assert_public_channel_rejects_others(channel: str) -> None:
    with pytest.raises(ValueError, match="forbidden channel"):
        wsmod._assert_public_channel(channel)


# 10. Forbidden channel rejection occurs before auth load.
def test_forbidden_channel_rejected_before_env_read(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "rec-id",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
    })
    fake_ws = FakeWS(frames=[])
    with pytest.raises(ValueError, match="forbidden channel"):
        wsmod.run(
            [
                "--ticker", "T",
                "--channel", "orderbook_delta",
                "--out", str(out),
                "--i-confirm-read-only-key",
            ],
            ws_factory=lambda kid, kp: fake_ws,
            env=env,
        )
    # Env must NOT have been queried — channel guard fires before auth load.
    assert env.gets == []
    # WS factory/start/subscribe must NOT have run.
    assert fake_ws.start_called is False
    assert fake_ws.subscribe_calls == []
    assert not out.exists()


# 11. Forbidden channel rejection occurs before ws.subscribe.
def test_forbidden_channel_rejected_before_ws_subscribe(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    fake_ws = FakeWS(frames=[])
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "rec-id",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
    })
    with pytest.raises(ValueError, match="forbidden channel"):
        wsmod.run(
            [
                "--ticker", "T",
                "--channel", "fill",
                "--out", str(out),
                "--i-confirm-read-only-key",
            ],
            ws_factory=lambda kid, kp: fake_ws,
            env=env,
        )
    assert fake_ws.subscribe_calls == []


# 12. Injected/fake WS cannot bypass _guarded_subscribe.
def test_guarded_subscribe_blocks_forbidden_channel_on_fake_ws() -> None:
    fake_ws = FakeWS(frames=[])

    async def _go() -> None:
        with pytest.raises(ValueError, match="forbidden channel"):
            await wsmod._guarded_subscribe(fake_ws, ["orderbook_delta"], ["T"])

    asyncio.run(_go())
    # The fake WS records subscribe calls; the guard must reject before any
    # call reaches it.
    assert fake_ws.subscribe_calls == []


def test_guarded_subscribe_passes_through_trade_channel() -> None:
    fake_ws = FakeWS(frames=[])

    async def _go() -> Any:
        return await wsmod._guarded_subscribe(fake_ws, ["trade"], ["T"])

    sub = asyncio.run(_go())
    assert hasattr(sub, "queue")
    assert fake_ws.subscribe_calls == [(("trade",), ("T",))]


# ---------------------------------------------------------------------------
# Runtime / output
# ---------------------------------------------------------------------------


# 13. --dry-run writes no output and does not open WebSocket.
def test_dry_run_does_not_open_ws_or_write_output(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    fake_ws = FakeWS(frames=[_trade_frame("T")])
    env = RecordingEnv({})  # no creds — must still succeed in dry-run
    rc = wsmod.run(
        [
            "--ticker", "T",
            "--out", str(out),
            "--dry-run",
        ],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
    )
    assert rc == 0
    assert not out.exists()
    assert fake_ws.start_called is False
    assert fake_ws.subscribe_calls == []
    assert fake_ws.stop_called is False
    # Dry-run must not require credentials — no env reads.
    assert env.gets == []


# 14. --ticker is required; no firehose.
def test_ticker_required_for_subscription(tmp_path: Path, capsys) -> None:
    out = tmp_path / "out.jsonl"
    fake_ws = FakeWS(frames=[])
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "x",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
    })
    rc = wsmod.run(
        ["--out", str(out), "--i-confirm-read-only-key"],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
    )
    assert rc != 0
    assert "ticker is required" in capsys.readouterr().err
    assert fake_ws.subscribe_calls == []
    assert not out.exists()


# 15. JSONL schema is stable.
def test_jsonl_record_schema_is_stable(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    frames = [_trade_frame("KXFED-26", seq=42, price=53), None]
    fake_ws = FakeWS(frames=frames)
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "rec-id",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
    })
    clock = MockClock()
    rc = wsmod.run(
        [
            "--ticker", "KXFED-26",
            "--max-duration-sec", "10",
            "--max-messages", "10",
            "--out", str(out),
            "--i-confirm-read-only-key",
        ],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
        clock_wall_ns=lambda: 9_000_000_000,
        clock_monotonic_ns=clock,
        async_sleep=_make_advancing_sleep(clock),
        session_id="fixed-session-id",
    )
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    expected_keys = {
        "source",
        "channel",
        "msg_type",
        "market_ticker",
        "captured_wall_ts_ns",
        "received_monotonic_ns",
        "session_id",
        "frame_seq",
        "kalshi_seq",
        "raw",
    }
    assert expected_keys <= set(rec.keys())
    assert rec["source"] == "kalshi_ws_readonly"
    assert rec["channel"] == "trade"
    assert rec["msg_type"] == "trade"
    assert rec["market_ticker"] == "KXFED-26"
    assert rec["session_id"] == "fixed-session-id"
    assert rec["frame_seq"] == 0
    assert rec["kalshi_seq"] == 42
    assert isinstance(rec["raw"], dict)
    assert rec["raw"]["msg"]["yes_price"] == 53


# 16. Append mode preserves existing data.
def test_append_preserves_existing_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    pre_existing = [
        {"source": "kalshi_ws_readonly", "frame_seq": 0, "ticker": "OLD"},
    ]
    out.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in pre_existing) + "\n",
        encoding="utf-8",
    )
    pre_text = out.read_text(encoding="utf-8")

    fake_ws = FakeWS(frames=[_trade_frame("NEW", seq=1), None])
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "rec-id",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
    })
    clock = MockClock()
    rc = wsmod.run(
        [
            "--ticker", "NEW",
            "--max-duration-sec", "10",
            "--max-messages", "10",
            "--out", str(out),
            "--i-confirm-read-only-key",
        ],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
        clock_wall_ns=lambda: 1,
        clock_monotonic_ns=clock,
        async_sleep=_make_advancing_sleep(clock),
        session_id="s",
    )
    assert rc == 0
    final_text = out.read_text(encoding="utf-8")
    assert final_text.startswith(pre_text)
    lines = final_text.splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["ticker"] == "OLD"
    assert json.loads(lines[1])["market_ticker"] == "NEW"


# 17. max-messages stops recorder.
def test_max_messages_stops_recorder(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    frames = [_trade_frame("T", seq=i) for i in range(50)]
    fake_ws = FakeWS(frames=frames)
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "x",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
    })
    clock = MockClock()
    rc = wsmod.run(
        [
            "--ticker", "T",
            "--max-duration-sec", "100",
            "--max-messages", "3",
            "--out", str(out),
            "--i-confirm-read-only-key",
        ],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
        clock_wall_ns=lambda: 1,
        clock_monotonic_ns=clock,
        async_sleep=_make_advancing_sleep(clock),
        session_id="s",
    )
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert fake_ws.stop_called is True


# 18. max-duration stops recorder.
def test_max_duration_stops_recorder(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    fake_ws = FakeWS(frames=[])  # never emits — only deadline can stop us
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "x",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
    })
    clock = MockClock()
    rc = wsmod.run(
        [
            "--ticker", "T",
            "--max-duration-sec", "0.1",
            "--max-messages", "1000",
            "--out", str(out),
            "--i-confirm-read-only-key",
        ],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
        clock_wall_ns=lambda: 1,
        clock_monotonic_ns=clock,
        async_sleep=_make_advancing_sleep(clock),
        session_id="s",
    )
    assert rc == 0
    # Empty queue + deadline expiration must produce no records.
    assert not out.exists() or out.read_text(encoding="utf-8") == ""
    assert fake_ws.stop_called is True


# 19. Malformed frame policy pinned: skip + increment malformed_count.
def test_malformed_frame_skipped_and_counted(tmp_path: Path, capsys) -> None:
    out = tmp_path / "out.jsonl"
    # Mix of malformed (str, int, list) and one legitimate frame, then sentinel.
    frames = [
        "this-is-not-a-frame",
        12345,
        ["not", "a", "dict"],
        _trade_frame("T", seq=7),
        None,
    ]
    fake_ws = FakeWS(frames=frames)
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "x",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
    })
    clock = MockClock()
    rc = wsmod.run(
        [
            "--ticker", "T",
            "--max-duration-sec", "10",
            "--max-messages", "10",
            "--out", str(out),
            "--i-confirm-read-only-key",
        ],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
        clock_wall_ns=lambda: 1,
        clock_monotonic_ns=clock,
        async_sleep=_make_advancing_sleep(clock),
        session_id="s",
    )
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["msg_type"] == "trade"
    summary_line = capsys.readouterr().out.strip().splitlines()[-1]
    summary = json.loads(summary_line)
    assert summary["records_written"] == 1
    assert summary["malformed_count"] == 3


# 20. Error frame policy pinned: write JSONL with error=true and raw payload.
def test_error_frame_recorded_with_error_flag(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    error_frame = {
        "type": "error",
        "code": 17,
        "msg": "rate limited or whatever",
    }
    frames = [error_frame, _trade_frame("T", seq=1), None]
    fake_ws = FakeWS(frames=frames)
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "x",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
    })
    clock = MockClock()
    rc = wsmod.run(
        [
            "--ticker", "T",
            "--max-duration-sec", "10",
            "--max-messages", "10",
            "--out", str(out),
            "--i-confirm-read-only-key",
        ],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
        clock_wall_ns=lambda: 1,
        clock_monotonic_ns=clock,
        async_sleep=_make_advancing_sleep(clock),
        session_id="s",
    )
    assert rc == 0
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    err_rec, trade_rec = lines
    assert err_rec.get("error") is True
    assert err_rec["msg_type"] == "error"
    assert err_rec["raw"] == error_frame
    assert trade_rec.get("error", False) is False
    assert trade_rec["msg_type"] == "trade"


def test_exception_frame_recorded_with_error_flag(tmp_path: Path) -> None:
    """Real KalshiWS maps server error frames to VenueError exceptions in
    the queue. The recorder must treat exception frames as error records."""
    out = tmp_path / "out.jsonl"

    class _FakeVenueError(Exception):
        pass

    frames: list[Any] = [_FakeVenueError("subscription_failed"), None]
    fake_ws = FakeWS(frames=frames)
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "x",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
    })
    clock = MockClock()
    rc = wsmod.run(
        [
            "--ticker", "T",
            "--max-duration-sec", "10",
            "--max-messages", "10",
            "--out", str(out),
            "--i-confirm-read-only-key",
        ],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
        clock_wall_ns=lambda: 1,
        clock_monotonic_ns=clock,
        async_sleep=_make_advancing_sleep(clock),
        session_id="s",
    )
    assert rc == 0
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["error"] is True
    assert rec["msg_type"] == "error"
    assert "subscription_failed" in rec["raw"]


# 21. No live WebSocket in tests.
def test_no_live_websocket_module_imported_during_tests() -> None:
    """If the recorder module ever pulled in 'websockets' at import time,
    this test would catch it. The default factory imports KalshiWS lazily."""
    # Re-import in subprocess to confirm the websockets package is not
    # loaded by simply importing the recorder.
    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %(repo)r)
        import executor.tools.kalshi_ws_snapshot  # noqa: F401
        print("websockets" in sys.modules)
        """
    ) % {"repo": str(REPO_ROOT)}
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "False", "recorder import must not load 'websockets' eagerly"


# ---------------------------------------------------------------------------
# Read-only acknowledgement gate
# ---------------------------------------------------------------------------


def test_non_dry_run_requires_read_only_flag(tmp_path: Path, capsys) -> None:
    out = tmp_path / "out.jsonl"
    fake_ws = FakeWS(frames=[])
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "x",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
    })
    rc = wsmod.run(
        ["--ticker", "T", "--out", str(out)],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
    )
    assert rc != 0
    assert not out.exists()
    # Auth must NOT have been queried — gate is BEFORE auth load.
    assert env.gets == []
    assert fake_ws.start_called is False
    err = capsys.readouterr().err
    assert "--i-confirm-read-only-key" in err


def test_read_only_warning_printed_for_non_dry_run(tmp_path: Path, capsys) -> None:
    out = tmp_path / "out.jsonl"
    fake_ws = FakeWS(frames=[None])
    env = RecordingEnv({
        "KALSHI_RECORDER_API_KEY_ID": "x",
        "KALSHI_RECORDER_PRIVATE_KEY_PATH": "/dev/null",
    })
    clock = MockClock()
    rc = wsmod.run(
        [
            "--ticker", "T",
            "--max-duration-sec", "5",
            "--max-messages", "5",
            "--out", str(out),
            "--i-confirm-read-only-key",
        ],
        ws_factory=lambda kid, kp: fake_ws,
        env=env,
        clock_wall_ns=lambda: 1,
        clock_monotonic_ns=clock,
        async_sleep=_make_advancing_sleep(clock),
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "read-only scope" in err


def test_dry_run_with_read_only_flag_does_not_print_warning(
    tmp_path: Path, capsys
) -> None:
    out = tmp_path / "out.jsonl"
    fake_ws = FakeWS(frames=[])
    rc = wsmod.run(
        [
            "--ticker", "T",
            "--out", str(out),
            "--dry-run",
            "--i-confirm-read-only-key",
        ],
        ws_factory=lambda kid, kp: fake_ws,
        env=RecordingEnv({}),
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" not in err


# ---------------------------------------------------------------------------
# Default path
# ---------------------------------------------------------------------------


def test_default_output_path_under_audit_logs_market_snapshots() -> None:
    p = wsmod.default_output_path()
    # Repo-relative — operator runs from /root/executor; CI runs from the
    # checked-out tree. Either way the JSONL lands under the repo's
    # audit-logs/ tree, never at an absolute system path.
    expected_parent = Path("audit-logs/market_snapshots/kalshi_ws")
    assert p.parent == expected_parent
    assert not p.is_absolute()
    assert p.suffix == ".jsonl"
    assert p.name.count("-") == 2  # YYYY-MM-DD.jsonl


# ---------------------------------------------------------------------------
# Allowlist constant
# ---------------------------------------------------------------------------


def test_allowed_ws_channels_is_trade_only_frozenset() -> None:
    assert isinstance(wsmod.ALLOWED_WS_CHANNELS, frozenset)
    assert wsmod.ALLOWED_WS_CHANNELS == frozenset({"trade"})
