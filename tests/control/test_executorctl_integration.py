"""executorctl end-to-end: subprocess CLI against a live ControlSocketServer."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

from executor.control.socket_server import ControlSocketServer
from executor.kill.manager import KillManager
from executor.kill.state import KillStateStore


EXECUTORCTL = Path(__file__).resolve().parents[2] / "scripts" / "executorctl"


async def _start(tmp_path: Path) -> tuple[ControlSocketServer, KillManager, Path]:
    store = KillStateStore(tmp_path / "kill.sqlite")
    mgr = KillManager(store=store, panic_cooldown_sec=1)
    sock = tmp_path / "ctl.sock"
    srv = ControlSocketServer(
        socket_path=str(sock),
        kill_mgr=mgr,
        daemon_started_ts_ns=time.time_ns(),
        git_sha="test-sha",
    )
    await srv.start()
    return srv, mgr, sock


def _run(sock: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["EXECUTOR_CONTROL_SOCKET"] = str(sock)
    return subprocess.run(
        [sys.executable, str(EXECUTORCTL), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


async def test_executorctl_ping_end_to_end(tmp_path: Path) -> None:
    srv, _, sock = await _start(tmp_path)
    try:
        loop = asyncio.get_running_loop()
        # Run subprocess in a worker thread so the event loop stays live
        # to serve the connection.
        result = await loop.run_in_executor(None, lambda: _run(sock, "ping"))
    finally:
        await srv.stop()
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "pong" in result.stdout


async def test_executorctl_kill_status_end_to_end(tmp_path: Path) -> None:
    srv, _, sock = await _start(tmp_path)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _run(sock, "kill", "status")
        )
    finally:
        await srv.stop()
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "mode=NONE" in result.stdout


def test_executorctl_connect_failure_exit_2(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.sock"
    result = _run(missing, "ping")
    assert result.returncode == 2
    assert "socket not found" in result.stderr or "daemon" in result.stderr


# ---------------------------------------------------------------------------
# Phase 4.14b — arm + heartbeat integration via the CLI against a live server
# wired with the dead-man surface.
# ---------------------------------------------------------------------------


async def _start_with_dead_man(tmp_path: Path):
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
            enabled=True,
            default_timeout_sec=600,
            min_timeout_sec=60,
            max_timeout_sec=3600,
        ),
    )
    sock = tmp_path / "ctl.sock"
    srv = ControlSocketServer(
        socket_path=str(sock),
        kill_mgr=mgr,
        daemon_started_ts_ns=time.time_ns(),
        operator_liveness=liveness,
        risk_config_getter=lambda: cfg,
    )
    await srv.start()
    return srv, sock


async def test_executorctl_arm_and_heartbeat_end_to_end(tmp_path: Path) -> None:
    srv, sock = await _start_with_dead_man(tmp_path)
    try:
        loop = asyncio.get_running_loop()
        arm_res = await loop.run_in_executor(
            None, lambda: _run(sock, "arm", "--timeout", "10m")
        )
        assert arm_res.returncode == 0, (
            f"stdout={arm_res.stdout!r} stderr={arm_res.stderr!r}"
        )
        assert "armed for 10m" in arm_res.stdout

        hb_res = await loop.run_in_executor(
            None, lambda: _run(sock, "heartbeat")
        )
        assert hb_res.returncode == 0, (
            f"stdout={hb_res.stdout!r} stderr={hb_res.stderr!r}"
        )
        assert "heartbeat ok" in hb_res.stdout

        status_res = await loop.run_in_executor(
            None, lambda: _run(sock, "arm_status")
        )
        assert status_res.returncode == 0
        assert "armed=True" in status_res.stdout
    finally:
        await srv.stop()
