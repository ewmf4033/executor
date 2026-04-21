"""
Operational tests for the weekly risk canary script.

These tests run canary.py as a subprocess with --dry-run, verifying:
1. The per_intent_dollar_cap gate rejects the oversized synthetic intent.
2. The script exits cleanly without crashing.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

VENV_PYTHON = "/root/executor/.venv/bin/python"
CANARY_SCRIPT = "/root/executor/scripts/canary.py"

# Common env: paper mode + fast durability, no Telegram.
_BASE_ENV = {
    **os.environ,
    "PAPER_MODE": "true",
    "EXECUTOR_AUDIT_DURABILITY": "NORMAL",
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
}


@pytest.mark.slow
def test_canary_detects_rejection(tmp_path: Path) -> None:
    """Default config has max_intent_dollars=250; canary sends $550.

    Expect exit 0 and output containing 'passed' or 'rejected'.
    """
    audit_dir = tmp_path / "audit"
    state_dir = tmp_path / "state"
    audit_dir.mkdir()
    state_dir.mkdir()

    result = subprocess.run(
        [
            VENV_PYTHON,
            CANARY_SCRIPT,
            "--dry-run",
            "--audit-dir", str(audit_dir),
            "--state-dir", str(state_dir),
        ],
        capture_output=True,
        text=True,
        env=_BASE_ENV,
        timeout=30,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"Canary exited {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    lower = combined.lower()
    assert "pass" in lower or "rejected" in lower, (
        f"Expected 'pass' or 'rejected' in output, got:\n{combined}"
    )


@pytest.mark.slow
def test_canary_runs_without_crash(tmp_path: Path) -> None:
    """Verify the canary script runs to completion in dry-run mode."""
    audit_dir = tmp_path / "audit"
    state_dir = tmp_path / "state"
    audit_dir.mkdir()
    state_dir.mkdir()

    result = subprocess.run(
        [
            VENV_PYTHON,
            CANARY_SCRIPT,
            "--dry-run",
            "--audit-dir", str(audit_dir),
            "--state-dir", str(state_dir),
        ],
        capture_output=True,
        text=True,
        env=_BASE_ENV,
        timeout=30,
    )
    # Exit 0 = gate working (pass), Exit 1 = gate bypassed (fail).
    # Both are "the script ran". Exit 2 = crash/error.
    assert result.returncode in (0, 1), (
        f"Canary crashed (exit {result.returncode}).\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Should not contain traceback from an unhandled exception.
    assert "Traceback (most recent call last)" not in result.stdout, (
        f"Unexpected traceback in stdout:\n{result.stdout}"
    )
