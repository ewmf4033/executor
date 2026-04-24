"""ControlSocketServer — roundtrip, permissions, dispatch, fault isolation."""
from __future__ import annotations

import asyncio
import json
import stat
import time
from pathlib import Path

import pytest

from executor.control.protocol import PROTOCOL_VERSION, encode
from executor.control.socket_server import ControlSocketServer
from executor.kill.manager import KillManager
from executor.kill.state import KillMode, KillStateStore


async def _start(tmp_path: Path) -> tuple[ControlSocketServer, KillManager, Path]:
    store = KillStateStore(tmp_path / "kill.sqlite")
    mgr = KillManager(store=store, panic_cooldown_sec=1)
    sock = tmp_path / "control.sock"
    srv = ControlSocketServer(
        socket_path=str(sock),
        kill_mgr=mgr,
        # Use a slightly past timestamp so uptime is non-zero.
        daemon_started_ts_ns=time.time_ns() - 1_500_000_000,
        git_sha="abcdef1",
    )
    await srv.start()
    return srv, mgr, sock


async def _roundtrip(sock: Path, request: dict) -> dict:
    reader, writer = await asyncio.open_unix_connection(str(sock))
    try:
        writer.write(encode(request))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    return json.loads(line.decode("utf-8"))


async def test_ping_returns_pong(tmp_path: Path) -> None:
    srv, _, sock = await _start(tmp_path)
    try:
        resp = await _roundtrip(
            sock, {"cmd": "ping", "args": {}, "source": "test"}
        )
    finally:
        await srv.stop()
    assert resp["ok"] is True
    assert resp["cmd"] == "ping"
    assert resp["result"]["pong"] is True
    assert resp["result"]["daemon_uptime_sec"] >= 1
    assert resp["result"]["protocol_version"] == PROTOCOL_VERSION


async def test_version_returns_metadata(tmp_path: Path) -> None:
    srv, _, sock = await _start(tmp_path)
    try:
        resp = await _roundtrip(
            sock, {"cmd": "version", "args": {}, "source": "test"}
        )
    finally:
        await srv.stop()
    assert resp["ok"] is True
    r = resp["result"]
    assert r["protocol_version"] == PROTOCOL_VERSION
    assert r["git_sha"] == "abcdef1"
    assert "executor_version" in r


async def test_kill_status_returns_current_mode(tmp_path: Path) -> None:
    srv, mgr, sock = await _start(tmp_path)
    try:
        r1 = await _roundtrip(
            sock, {"cmd": "kill_status", "args": {}, "source": "test"}
        )
        assert r1["ok"] is True
        assert r1["result"]["mode"] == "NONE"
        # None of the internal fields should leak.
        assert "engaged_ts_ns" not in r1["result"]
        assert "last_resume_ts_ns" not in r1["result"]
        assert "extra" not in r1["result"]

        await mgr.engage(KillMode.SOFT, "manual")
        r2 = await _roundtrip(
            sock, {"cmd": "kill_status", "args": {}, "source": "test"}
        )
        assert r2["result"]["mode"] == "SOFT"
        assert r2["result"]["reason"] == "manual"
    finally:
        await srv.stop()


async def test_kill_soft_engages_soft(tmp_path: Path) -> None:
    srv, mgr, sock = await _start(tmp_path)
    try:
        resp = await _roundtrip(
            sock,
            {
                "cmd": "kill",
                "args": {"sub": "soft", "reason": "t-soft"},
                "source": "test",
            },
        )
    finally:
        await srv.stop()
    assert resp["ok"] is True
    assert resp["result"]["mode"] == "SOFT"
    assert resp["result"]["source"] == "executorctl"
    assert mgr.snapshot().mode == KillMode.SOFT
    assert mgr.snapshot().reason == "t-soft"


async def test_kill_hard_engages_hard_with_cancel_open(tmp_path: Path) -> None:
    srv, mgr, sock = await _start(tmp_path)
    try:
        resp = await _roundtrip(
            sock,
            {
                "cmd": "kill",
                "args": {"sub": "hard", "reason": "t-hard"},
                "source": "test",
            },
        )
    finally:
        await srv.stop()
    assert resp["ok"] is True
    assert resp["result"]["mode"] == "HARD"
    # Verify hard mode took effect on the shared KillSwitch as well.
    assert mgr.snapshot().mode == KillMode.HARD
    killed, _ = mgr.is_killed()
    assert killed is True


async def test_kill_panic_engages_panic_hard(tmp_path: Path) -> None:
    srv, mgr, sock = await _start(tmp_path)
    try:
        resp = await _roundtrip(
            sock,
            {
                "cmd": "kill",
                "args": {"sub": "panic", "reason": "t-panic"},
                "source": "test",
            },
        )
    finally:
        await srv.stop()
    assert resp["ok"] is True
    assert resp["result"]["mode"] == "HARD"
    assert resp["result"]["panic"] is True
    snap = mgr.snapshot()
    assert snap.panic is True
    assert snap.manual_only is True


async def test_kill_resume_clears_state(tmp_path: Path) -> None:
    srv, mgr, sock = await _start(tmp_path)
    try:
        await mgr.engage(KillMode.SOFT, "pre")
        resp = await _roundtrip(
            sock,
            {"cmd": "kill", "args": {"sub": "resume"}, "source": "test"},
        )
    finally:
        await srv.stop()
    assert resp["ok"] is True
    assert resp["result"]["mode"] == "NONE"
    assert mgr.snapshot().mode == KillMode.NONE


async def test_kill_rejects_missing_reason(tmp_path: Path) -> None:
    srv, mgr, sock = await _start(tmp_path)
    try:
        resp = await _roundtrip(
            sock,
            {"cmd": "kill", "args": {"sub": "soft"}, "source": "test"},
        )
    finally:
        await srv.stop()
    assert resp["ok"] is False
    assert resp["code"] == "invalid_args"
    assert mgr.snapshot().mode == KillMode.NONE


async def test_kill_rejects_invalid_sub(tmp_path: Path) -> None:
    srv, _, sock = await _start(tmp_path)
    try:
        resp = await _roundtrip(
            sock,
            {
                "cmd": "kill",
                "args": {"sub": "nuke", "reason": "x"},
                "source": "test",
            },
        )
    finally:
        await srv.stop()
    assert resp["ok"] is False
    assert resp["code"] == "invalid_args"


async def test_unknown_command_returns_error(tmp_path: Path) -> None:
    srv, _, sock = await _start(tmp_path)
    try:
        resp = await _roundtrip(
            sock,
            {"cmd": "fly_me_to_mars", "args": {}, "source": "test"},
        )
    finally:
        await srv.stop()
    assert resp["ok"] is False
    assert resp["code"] == "unknown_command"


async def test_malformed_request_returns_error(tmp_path: Path) -> None:
    srv, _, sock = await _start(tmp_path)
    try:
        reader, writer = await asyncio.open_unix_connection(str(sock))
        writer.write(b"garbage not json\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    finally:
        await srv.stop()
    resp = json.loads(line.decode("utf-8"))
    assert resp["ok"] is False
    assert resp["code"] == "invalid_args"


async def test_socket_permissions_0600_after_start(tmp_path: Path) -> None:
    srv, _, sock = await _start(tmp_path)
    try:
        mode = sock.stat().st_mode
        assert stat.S_IMODE(mode) == 0o600
    finally:
        await srv.stop()


async def test_stale_socket_file_removed_on_start(tmp_path: Path) -> None:
    sock = tmp_path / "control.sock"
    sock.write_text("stale leftover")
    assert sock.exists() and sock.is_file()
    store = KillStateStore(tmp_path / "kill.sqlite")
    mgr = KillManager(store=store, panic_cooldown_sec=1)
    srv = ControlSocketServer(
        socket_path=str(sock),
        kill_mgr=mgr,
        daemon_started_ts_ns=time.time_ns(),
    )
    await srv.start()
    try:
        # After start, sock exists but is now a socket (not a regular file).
        assert sock.exists()
        resp = await _roundtrip(sock, {"cmd": "ping", "args": {}, "source": "t"})
        assert resp["ok"] is True
    finally:
        await srv.stop()


async def test_one_bad_client_does_not_crash_server(tmp_path: Path) -> None:
    srv, _, sock = await _start(tmp_path)
    try:
        # Bad client 1: malformed request.
        reader, writer = await asyncio.open_unix_connection(str(sock))
        writer.write(b"garbage\n")
        await writer.drain()
        _ = await asyncio.wait_for(reader.readline(), timeout=2.0)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        # Bad client 2: connect then close without sending anything.
        _, w2 = await asyncio.open_unix_connection(str(sock))
        w2.close()
        try:
            await w2.wait_closed()
        except Exception:
            pass
        # Valid client afterwards must still succeed.
        resp = await _roundtrip(sock, {"cmd": "ping", "args": {}, "source": "t"})
        assert resp["ok"] is True
    finally:
        await srv.stop()
