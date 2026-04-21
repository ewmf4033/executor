"""
Operational tests for the daily digest script.

Creates temporary audit and attribution DBs, inserts test events, and
runs daily_digest.py --dry-run to verify output and saved JSON.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

VENV_PYTHON = "/root/executor/.venv/bin/python"
DIGEST_SCRIPT = "/root/executor/scripts/daily_digest.py"

_BASE_ENV = {
    **os.environ,
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
}


def _create_audit_db(db_path: Path, target_date: dt.date) -> None:
    """Create an audit DB and insert test events for the given date."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            ts_ns INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            source TEXT NOT NULL,
            intent_id TEXT,
            leg_id TEXT,
            venue TEXT,
            market_id TEXT,
            strategy_id TEXT,
            payload_json TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1
        )
    """)

    # Place events in the middle of target_date (noon UTC)
    base_ts = int(dt.datetime(
        target_date.year, target_date.month, target_date.day,
        12, 0, 0, tzinfo=dt.timezone.utc,
    ).timestamp() * 1e9)

    events = [
        # 3 INTENT_EMITTED
        ("INTENT_EMITTED", "strategy:test", "intent-1", "{}"),
        ("INTENT_EMITTED", "strategy:test", "intent-2", "{}"),
        ("INTENT_EMITTED", "strategy:test", "intent-3", "{}"),
        # 2 INTENT_ADMITTED
        ("INTENT_ADMITTED", "orchestrator", "intent-1", "{}"),
        ("INTENT_ADMITTED", "orchestrator", "intent-2", "{}"),
        # 2 GATE_REJECTED (one structural, one zscore)
        ("GATE_REJECTED", "risk_policy", "intent-3",
         json.dumps({"gate": "structural", "reason": "test"})),
        ("GATE_REJECTED", "risk_policy", "intent-3",
         json.dumps({"gate": "zscore", "reason": "test"})),
        # 1 FILL
        ("FILL", "venue_gw", "intent-1", json.dumps({"venue": "paper", "size": "0.1"})),
        # 1 SELF_CHECK_OK
        ("SELF_CHECK_OK", "self_check", None, "{}"),
        # 1 ERROR
        ("ERROR", "venue_gw", None, json.dumps({"msg": "timeout"})),
    ]

    for i, (etype, source, intent_id, payload) in enumerate(events):
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, 1)",
            (str(uuid.uuid4()), base_ts + i * 1_000_000, etype, source,
             intent_id, "test", payload),
        )

    conn.commit()
    conn.close()


def _create_attribution_db(db_path: Path, target_date: dt.date) -> None:
    """Create an attribution DB with 2 test fills."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE attribution (
            fill_id TEXT PRIMARY KEY,
            intent_id TEXT NOT NULL,
            leg_id TEXT NOT NULL,
            strategy_id TEXT NOT NULL,
            venue TEXT NOT NULL,
            market_id TEXT NOT NULL,
            side TEXT NOT NULL,
            size TEXT NOT NULL,
            intent_price TEXT,
            decision_price TEXT,
            arrival_price TEXT,
            fill_price TEXT NOT NULL,
            exit_price TEXT,
            strategy_edge TEXT,
            execution_cost TEXT,
            short_term_alpha TEXT,
            fee TEXT,
            fill_ts_ns INTEGER NOT NULL,
            settled_ts_ns INTEGER NOT NULL,
            extra_json TEXT NOT NULL DEFAULT '{}'
        )
    """)

    base_ts = int(dt.datetime(
        target_date.year, target_date.month, target_date.day,
        12, 0, 0, tzinfo=dt.timezone.utc,
    ).timestamp() * 1e9)

    fills = [
        (str(uuid.uuid4()), "intent-1", "leg-1", "test_strat", "paper",
         "BTC-PERP", "buy", "0.1", "50000", "50000", "50000", "50010",
         None, None, None, "0.0005", "0.05",
         base_ts, base_ts + 1_000_000),
        (str(uuid.uuid4()), "intent-2", "leg-2", "test_strat", "paper",
         "ETH-PERP", "sell", "1.0", "3000", "3000", "3000", "2998",
         None, None, None, "0.001", "0.03",
         base_ts + 2_000_000, base_ts + 3_000_000),
    ]

    for f in fills:
        conn.execute(
            "INSERT INTO attribution VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')",
            f,
        )

    conn.commit()
    conn.close()


def _run_digest(tmp_path: Path, target_date: dt.date) -> subprocess.CompletedProcess:
    """Set up tmp DBs and run digest --dry-run."""
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    digest_dir = tmp_path / "digests"
    digest_dir.mkdir()
    attr_db = tmp_path / "attribution.sqlite"

    _create_audit_db(audit_dir / f"audit-{target_date.isoformat()}.sqlite", target_date)
    _create_attribution_db(attr_db, target_date)

    result = subprocess.run(
        [
            VENV_PYTHON, DIGEST_SCRIPT,
            "--dry-run",
            "--audit-dir", str(audit_dir),
            "--attr-db", str(attr_db),
            "--digest-dir", str(digest_dir),
            "--date", target_date.isoformat(),
        ],
        capture_output=True, text=True, timeout=30,
        env=_BASE_ENV,
    )
    return result


@pytest.mark.slow
def test_digest_counts_events(tmp_path: Path) -> None:
    """Verify event counts match the inserted test data."""
    target_date = dt.date.today() - dt.timedelta(days=1)
    result = _run_digest(tmp_path, target_date)

    assert result.returncode == 0, f"Script failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    out = result.stdout

    assert "Digest" in out
    # Pipeline counts: 3 emitted, 2 admitted, 2 rejected, 1 error
    assert "Emitted: 3" in out
    assert "Admitted: 2" in out
    assert "Rejected: 2" in out
    assert "Crashed: 1" in out
    # Fill count from attribution DB
    assert "Count: 2" in out
    # Self-check
    assert "OK" in out


@pytest.mark.slow
def test_digest_under_4000_chars(tmp_path: Path) -> None:
    """Verify the formatted message fits in Telegram's 4096 char limit."""
    target_date = dt.date.today() - dt.timedelta(days=1)
    result = _run_digest(tmp_path, target_date)

    assert result.returncode == 0
    # Extract the message between the === separators
    lines = result.stdout.split("\n")
    in_msg = False
    msg_lines = []
    for line in lines:
        if line.startswith("=" * 20):
            if in_msg:
                break
            in_msg = True
            continue
        if in_msg:
            msg_lines.append(line)
    message = "\n".join(msg_lines)
    assert len(message) < 4000, f"Message is {len(message)} chars, exceeds 4000"


@pytest.mark.slow
def test_digest_saves_json(tmp_path: Path) -> None:
    """Verify a digest JSON file is created with expected keys."""
    target_date = dt.date.today() - dt.timedelta(days=1)
    digest_dir = tmp_path / "digests"

    result = _run_digest(tmp_path, target_date)
    assert result.returncode == 0

    json_path = digest_dir / f"digest-{target_date.isoformat()}.json"
    assert json_path.exists(), f"Expected digest JSON at {json_path}"

    with open(json_path) as f:
        data = json.load(f)

    assert "event_counts" in data
    assert "fill_count" in data
    assert "target_date" in data
    assert data["target_date"] == target_date.isoformat()
    assert data["event_counts"]["INTENT_EMITTED"] == 3
    assert data["fill_count"] == 2
