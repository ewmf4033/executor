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


# ---------------------------------------------------------------------------
# Phase 4.14b — dead-man (arm/disarm/heartbeat/arm_status) handlers.
# ---------------------------------------------------------------------------


async def _start_with_dead_man(
    tmp_path: Path, *, enabled: bool = True
) -> tuple[ControlSocketServer, Path]:
    from dataclasses import replace as _replace

    from executor.risk.config import DeadManCfg, RiskConfig
    from executor.risk.state import OperatorLivenessStore, RiskState

    store = KillStateStore(tmp_path / "kill.sqlite")
    mgr = KillManager(store=store, panic_cooldown_sec=1)
    rs = RiskState(db_path=tmp_path / "risk_state.sqlite")
    await rs.load()
    liveness = OperatorLivenessStore(rs.connection)
    cfg = _replace(
        RiskConfig(),
        dead_man=DeadManCfg(
            enabled=enabled,
            default_timeout_sec=600,
            min_timeout_sec=60,
            max_timeout_sec=3600,
        ),
    )
    sock = tmp_path / "control.sock"
    srv = ControlSocketServer(
        socket_path=str(sock),
        kill_mgr=mgr,
        daemon_started_ts_ns=time.time_ns(),
        operator_liveness=liveness,
        risk_config_getter=lambda: cfg,
    )
    await srv.start()
    return srv, sock


async def test_arm_engages_when_enabled(tmp_path: Path) -> None:
    srv, sock = await _start_with_dead_man(tmp_path, enabled=True)
    try:
        resp = await _roundtrip(
            sock,
            {"cmd": "arm", "args": {"timeout_sec": 600}, "source": "test"},
        )
    finally:
        await srv.stop()
    assert resp["ok"] is True
    assert resp["result"]["armed"] is True
    assert resp["result"]["timeout_sec"] == 600
    assert resp["result"]["kill_mode_at_arm"] == "NONE"


async def test_arm_rejects_when_disabled(tmp_path: Path) -> None:
    srv, sock = await _start_with_dead_man(tmp_path, enabled=False)
    try:
        resp = await _roundtrip(
            sock,
            {"cmd": "arm", "args": {"timeout_sec": 600}, "source": "test"},
        )
    finally:
        await srv.stop()
    assert resp["ok"] is False
    assert "dead_man_disabled" in resp["error"]


async def test_heartbeat_updates_last_hb(tmp_path: Path) -> None:
    srv, sock = await _start_with_dead_man(tmp_path, enabled=True)
    try:
        await _roundtrip(
            sock,
            {"cmd": "arm", "args": {"timeout_sec": 600}, "source": "test"},
        )
        resp = await _roundtrip(
            sock, {"cmd": "heartbeat", "args": {}, "source": "test"}
        )
        status = await _roundtrip(
            sock, {"cmd": "arm_status", "args": {}, "source": "test"}
        )
    finally:
        await srv.stop()
    assert resp["ok"] is True
    assert resp["result"]["armed"] is True
    assert resp["result"]["seconds_until_stale"] > 0
    assert status["ok"] is True
    assert status["result"]["armed"] is True


async def test_disarm_clears_armed(tmp_path: Path) -> None:
    srv, sock = await _start_with_dead_man(tmp_path, enabled=True)
    try:
        await _roundtrip(
            sock,
            {"cmd": "arm", "args": {"timeout_sec": 600}, "source": "test"},
        )
        resp = await _roundtrip(
            sock,
            {"cmd": "disarm", "args": {"reason": "eod"}, "source": "test"},
        )
        status = await _roundtrip(
            sock, {"cmd": "arm_status", "args": {}, "source": "test"}
        )
    finally:
        await srv.stop()
    assert resp["ok"] is True
    assert resp["result"]["armed"] is False
    assert resp["result"]["disarmed_reason"] == "eod"
    assert status["result"]["armed"] is False


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


# ---------------------------------------------------------------------------
# Phase 4.14e — runtime dir + socket hardening (H2, H3).
#
# The ultrareview flagged the fallback parent-dir mkdir mode (0o755) and
# the bind→chmod TOCTOU: ``asyncio.start_unix_server`` creates the inode
# honoring the current umask, which was still 0o022 on typical systems,
# leaving a brief window where the socket was world-readable before our
# explicit chmod 0o600 ran. We now umask(0o177) around the bind and
# tighten the mkdir fallback to 0o700.
# ---------------------------------------------------------------------------


async def test_socket_parent_dir_mode_0700_fallback(tmp_path: Path) -> None:
    """Fallback mkdir of the socket parent creates a 0700 directory."""
    parent = tmp_path / "run-exec-nonexistent"
    sock = parent / "control.sock"
    assert not parent.exists()
    store = KillStateStore(tmp_path / "kill.sqlite")
    mgr = KillManager(store=store, panic_cooldown_sec=1)
    srv = ControlSocketServer(
        socket_path=str(sock),
        kill_mgr=mgr,
        daemon_started_ts_ns=time.time_ns(),
    )
    await srv.start()
    try:
        assert parent.exists()
        # Parent dir was created by our fallback; mode must be 0o700.
        mode = stat.S_IMODE(parent.stat().st_mode)
        assert mode == 0o700, f"expected 0o700, got {oct(mode)}"
    finally:
        await srv.stop()


async def test_socket_file_mode_0600(tmp_path: Path) -> None:
    """Socket file mode must be exactly 0o600 regardless of process umask.

    This specifically exercises the TOCTOU fix: we set the process umask
    to a permissive 0o000 before ``srv.start()`` so, without the fix,
    ``start_unix_server`` would create the socket at 0o666 until chmod
    caught up. With umask(0o177) wrapping the bind, the socket is 0o600
    at creation time.
    """
    import os as _os

    store = KillStateStore(tmp_path / "kill.sqlite")
    mgr = KillManager(store=store, panic_cooldown_sec=1)
    sock = tmp_path / "control.sock"
    srv = ControlSocketServer(
        socket_path=str(sock),
        kill_mgr=mgr,
        daemon_started_ts_ns=time.time_ns(),
    )
    original_umask = _os.umask(0o000)
    try:
        await srv.start()
        try:
            mode = stat.S_IMODE(sock.stat().st_mode)
            assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
        finally:
            await srv.stop()
    finally:
        _os.umask(original_umask)


# ---------------------------------------------------------------------------
# Phase 4.14e (M8) — handler-cancellation shielding for hard/panic
# engage. ``asyncio.shield`` wraps the ``KillManager.engage`` call so
# that a cancellation of the outer handler coroutine (client
# disconnect, server.close() sweeping in-flight handlers, shutdown
# cancelling the handler task) does not abort the engage mid-way
# through its state persistence. shield only rescues the engage
# coroutine from *caller-side* cancellation — DB or lock stalls are
# out of scope.
# ---------------------------------------------------------------------------


async def test_panic_engage_is_shielded_from_handler_cancellation(
    tmp_path: Path,
) -> None:
    """If we cancel the _dispatch task mid-await while ``engage`` is
    running, the engage must still complete and persist panic state."""
    from executor.kill.state import KillMode

    store = KillStateStore(tmp_path / "kill.sqlite")
    mgr = KillManager(store=store, panic_cooldown_sec=1)
    sock = tmp_path / "control.sock"
    srv = ControlSocketServer(
        socket_path=str(sock),
        kill_mgr=mgr,
        daemon_started_ts_ns=time.time_ns(),
    )
    # Swap in a slow engage that yields, so we can cancel the caller
    # mid-engage and observe the shield protecting the inner work.
    original_engage = mgr.engage
    started = asyncio.Event()
    finished = asyncio.Event()

    async def _slow_engage(*args, **kwargs):
        started.set()
        # Yield and sleep briefly to simulate persistence time.
        await asyncio.sleep(0.05)
        result = await original_engage(*args, **kwargs)
        finished.set()
        return result

    mgr.engage = _slow_engage  # type: ignore[assignment]

    task = asyncio.create_task(
        srv._do_kill({"sub": "panic", "reason": "boom"})
    )
    await started.wait()
    task.cancel()
    # Swallow the cancel — the outer handler is cancelled but the
    # engage must have been shielded.
    try:
        await task
    except asyncio.CancelledError:
        pass
    # The inner engage completes regardless of the outer cancellation.
    await asyncio.wait_for(finished.wait(), timeout=2.0)
    snap = mgr.snapshot()
    assert snap.mode == KillMode.HARD
    assert snap.panic is True


async def test_hard_engage_is_shielded_from_handler_cancellation(
    tmp_path: Path,
) -> None:
    from executor.kill.state import KillMode

    store = KillStateStore(tmp_path / "kill.sqlite")
    mgr = KillManager(store=store, panic_cooldown_sec=1)
    sock = tmp_path / "control.sock"
    srv = ControlSocketServer(
        socket_path=str(sock),
        kill_mgr=mgr,
        daemon_started_ts_ns=time.time_ns(),
    )
    original_engage = mgr.engage
    started = asyncio.Event()
    finished = asyncio.Event()

    async def _slow_engage(*args, **kwargs):
        started.set()
        await asyncio.sleep(0.05)
        result = await original_engage(*args, **kwargs)
        finished.set()
        return result

    mgr.engage = _slow_engage  # type: ignore[assignment]
    task = asyncio.create_task(
        srv._do_kill({"sub": "hard", "reason": "halt"})
    )
    await started.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await asyncio.wait_for(finished.wait(), timeout=2.0)
    assert mgr.snapshot().mode == KillMode.HARD


async def test_shielded_engage_exception_after_cancel_is_logged(
    tmp_path: Path,
) -> None:
    """Phase 4.14e follow-up: if the outer handler is cancelled AND
    the shielded inner engage later raises, the exception must NOT be
    silently lost. We replace engage with a coroutine that raises after
    a short sleep; cancel the outer task while it's mid-engage; then
    verify the inner task's exception is retrieved via the
    done-callback (no ``Task exception was never retrieved`` warning)
    and surfaced to the module's error-log hook."""
    from executor.control import socket_server as socket_server_mod

    store = KillStateStore(tmp_path / "kill.sqlite")
    mgr = KillManager(store=store, panic_cooldown_sec=1)
    srv = socket_server_mod.ControlSocketServer(
        socket_path=str(tmp_path / "control.sock"),
        kill_mgr=mgr,
        daemon_started_ts_ns=time.time_ns(),
    )
    started = asyncio.Event()

    async def _raising_engage(*_a, **_kw):
        started.set()
        await asyncio.sleep(0.05)
        raise RuntimeError("engage-blew-up-after-cancel")

    mgr.engage = _raising_engage  # type: ignore[assignment]

    # Patch the module logger's .error so we can observe the retrieval.
    captured: list[tuple[str, dict]] = []
    original_error = socket_server_mod.log.error

    def _spy_error(event, **kw):
        captured.append((event, kw))
        return original_error(event, **kw)

    socket_server_mod.log.error = _spy_error  # type: ignore[assignment]
    try:
        task = asyncio.create_task(
            srv._do_kill({"sub": "panic", "reason": "boom"})
        )
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Allow the inner task's done-callback to fire.
        await asyncio.sleep(0.15)
    finally:
        socket_server_mod.log.error = original_error  # type: ignore[assignment]

    # The error must have been retrieved + logged — not silently lost.
    assert any(
        evt == "control.socket.shielded_engage_failed_after_cancel"
        and kw.get("kind") == "panic"
        and "engage-blew-up-after-cancel" in kw.get("error", "")
        for evt, kw in captured
    ), captured


async def test_stale_socket_unlink_handles_missing_file(tmp_path: Path) -> None:
    """No stale socket present — start must succeed without stat races."""
    store = KillStateStore(tmp_path / "kill.sqlite")
    mgr = KillManager(store=store, panic_cooldown_sec=1)
    sock = tmp_path / "missing-before-start.sock"
    assert not sock.exists()
    srv = ControlSocketServer(
        socket_path=str(sock),
        kill_mgr=mgr,
        daemon_started_ts_ns=time.time_ns(),
    )
    await srv.start()
    try:
        resp = await _roundtrip(sock, {"cmd": "ping", "args": {}, "source": "t"})
        assert resp["ok"] is True
    finally:
        await srv.stop()
        assert not sock.exists(), "stop() must also unlink without a stat race"


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
