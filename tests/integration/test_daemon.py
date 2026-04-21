"""Integration: DaemonService end-to-end + SIGTERM graceful shutdown."""
from __future__ import annotations

import asyncio
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from executor.core.daemon import DaemonService, run_daemon


@pytest.mark.asyncio
async def test_daemon_self_check_only_returns_zero(tmp_path: Path):
    os.environ["PAPER_MODE"] = "true"
    rc = await run_daemon(
        self_check_only=True,
        audit_dir=tmp_path / "audit",
        risk_yaml="/root/executor/config/risk.yaml",
        risk_state_db=tmp_path / "rstate.sqlite",
        kill_db=tmp_path / "kill.sqlite",
        attribution_db=tmp_path / "attr.sqlite",
        telemetry_port=0,
        enable_quote_feeder=False,
    )
    assert rc == 0
    # Audit DB should contain SELF_CHECK_OK + the full pipeline.
    audit_files = list((tmp_path / "audit").glob("audit-*.sqlite"))
    assert len(audit_files) == 1
    conn = sqlite3.connect(str(audit_files[0]))
    try:
        rows = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT event_type, COUNT(*) FROM events GROUP BY event_type"
            )
        }
    finally:
        conn.close()
    # Every stage of the pipeline must have fired.
    for needed in (
        "INTENT_EMITTED",
        "INTENT_ADMITTED",
        "ORDER_PLACED",
        "FILL",
        "INTENT_COMPLETE",
        "SELF_CHECK_OK",
    ):
        assert rows.get(needed, 0) >= 1, f"missing {needed}: {rows}"


def _subprocess_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PAPER_MODE"] = "true"
    env["EXECUTOR_AUDIT_DIR"] = str(tmp_path / "audit")
    return env


@pytest.mark.asyncio
async def test_daemon_sigterm_writes_state_saved(tmp_path: Path):
    """Start the daemon as a subprocess, let it run through self-check, send
    SIGTERM, verify STATE_SAVED hit the audit log."""
    env = _subprocess_env(tmp_path)
    cmd = [
        sys.executable, "-m", "executor",
        "--daemon",
        "--audit-dir", str(tmp_path / "audit"),
        "--telemetry-port", "0",
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd="/root/executor",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        # Wait up to 10s for the audit DB to appear and SELF_CHECK_OK to be written.
        deadline = time.time() + 10.0
        ok_seen = False
        audit_file: Path | None = None
        while time.time() < deadline:
            audit_files = list((tmp_path / "audit").glob("audit-*.sqlite"))
            if audit_files:
                audit_file = audit_files[0]
                try:
                    c = sqlite3.connect(str(audit_file))
                    n = c.execute(
                        "SELECT COUNT(*) FROM events WHERE event_type='SELF_CHECK_OK'"
                    ).fetchone()[0]
                    c.close()
                    if n >= 1:
                        ok_seen = True
                        break
                except sqlite3.DatabaseError:
                    pass
            await asyncio.sleep(0.25)
        assert ok_seen, "daemon never wrote SELF_CHECK_OK"
        # Graceful shutdown.
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
        assert proc.returncode == 0, f"daemon exited non-zero: {proc.returncode}"
        # STATE_SAVED must be in the audit log.
        assert audit_file is not None
        c = sqlite3.connect(str(audit_file))
        try:
            n = c.execute(
                "SELECT COUNT(*) FROM events WHERE event_type='STATE_SAVED'"
            ).fetchone()[0]
        finally:
            c.close()
        assert n >= 1, "SIGTERM did not produce STATE_SAVED"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.asyncio
async def test_daemon_refuses_without_paper_mode(tmp_path: Path, monkeypatch):
    """PAPER_MODE=false makes DaemonService raise at construction — belt for the CLI lock."""
    monkeypatch.setenv("PAPER_MODE", "false")
    with pytest.raises(RuntimeError, match="PAPER_MODE=true"):
        DaemonService(
            audit_dir=tmp_path / "audit",
            risk_yaml="/root/executor/config/risk.yaml",
            risk_state_db=tmp_path / "rstate.sqlite",
            kill_db=tmp_path / "kill.sqlite",
            attribution_db=tmp_path / "attr.sqlite",
            telemetry_port=0,
        )
