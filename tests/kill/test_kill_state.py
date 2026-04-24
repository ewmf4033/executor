"""Persistence round-trip for KillStateStore."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from executor.kill.state import (
    FORCE_RESET_ENV_VAR,
    KillMode,
    KillStateSnapshot,
    KillStateStore,
)


def test_default_snapshot_when_empty(tmp_path: Path) -> None:
    store = KillStateStore(tmp_path / "kill.sqlite")
    snap = store.load()
    assert snap.mode is KillMode.NONE
    assert snap.reason == ""
    assert snap.panic is False
    assert snap.resume_strikes == 0
    assert snap.manual_only is False
    store.close()


def test_save_and_reload_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "kill.sqlite"
    store = KillStateStore(db)
    snap = KillStateSnapshot(
        mode=KillMode.HARD,
        reason="manual panic",
        engaged_ts_ns=12345,
        panic=True,
        panic_until_ns=99999,
        resume_strikes=2,
        manual_only=True,
        extra={"who": "ari"},
    )
    store.save(snap)
    store.close()

    store2 = KillStateStore(db)
    out = store2.load()
    assert out.mode is KillMode.HARD
    assert out.reason == "manual panic"
    assert out.engaged_ts_ns == 12345
    assert out.panic is True
    assert out.panic_until_ns == 99999
    assert out.resume_strikes == 2
    assert out.manual_only is True
    assert out.extra == {"who": "ari"}
    store2.close()


def test_save_overwrites_single_row(tmp_path: Path) -> None:
    store = KillStateStore(tmp_path / "kill.sqlite")
    store.save(KillStateSnapshot(mode=KillMode.SOFT, reason="r1", engaged_ts_ns=1))
    store.save(KillStateSnapshot(mode=KillMode.HARD, reason="r2", engaged_ts_ns=2))
    out = store.load()
    assert out.mode is KillMode.HARD
    assert out.reason == "r2"
    store.close()


def test_kill_mode_enum_values() -> None:
    assert KillMode.NONE.value == "NONE"
    assert KillMode.SOFT.value == "SOFT"
    assert KillMode.HARD.value == "HARD"


# --------------------------------------------------------------------------
# Phase 4.13 — fail-closed corruption rebuild
# --------------------------------------------------------------------------


def test_corrupt_kill_db_rebuilds_to_hard_manual_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the bypass env var, a corrupt kill DB must rebuild into
    HARD + manual_only with reason=KILL_DB_CORRUPT_REBUILT — trading stays
    stopped until an operator resolves manually."""
    monkeypatch.delenv(FORCE_RESET_ENV_VAR, raising=False)
    db_path = tmp_path / "kill_state.sqlite"
    db_path.write_bytes(b"not a sqlite database")

    store = KillStateStore(db_path)
    try:
        assert store.rebuilt_from_corruption is True
        assert store.force_reset_used is False
        snap = store.load()
        assert snap.mode is KillMode.HARD
        assert snap.manual_only is True
        assert snap.reason == "KILL_DB_CORRUPT_REBUILT"
        assert snap.engaged_ts_ns > 0
        assert snap.extra.get("rebuilt_from_corruption") is True
    finally:
        store.close()


def test_corrupt_kill_db_with_force_reset_rebuilds_to_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With EXECUTOR_FORCE_RESET_KILL_STATE=1, the rebuild seeds NONE and
    the store flips force_reset_used=True so the daemon startup path can
    emit KILL_STATE_FORCE_RESET."""
    monkeypatch.setenv(FORCE_RESET_ENV_VAR, "1")
    db_path = tmp_path / "kill_state.sqlite"
    db_path.write_bytes(b"garbage bytes, not sqlite")

    store = KillStateStore(db_path)
    try:
        assert store.rebuilt_from_corruption is True
        assert store.force_reset_used is True
        snap = store.load()
        assert snap.mode is KillMode.NONE
        assert snap.manual_only is False
        assert snap.reason == "KILL_DB_CORRUPT_REBUILT_FORCE_RESET"
        assert snap.extra.get("force_reset_used") is True
    finally:
        store.close()


def test_corrupt_kill_db_force_reset_not_honored_when_env_value_wrong(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the exact value "1" honors the bypass — other truthy values
    must fail-closed to prevent accidental typos from opening the door."""
    monkeypatch.setenv(FORCE_RESET_ENV_VAR, "true")
    db_path = tmp_path / "kill_state.sqlite"
    db_path.write_bytes(b"garbage")

    store = KillStateStore(db_path)
    try:
        assert store.force_reset_used is False
        snap = store.load()
        assert snap.mode is KillMode.HARD
        assert snap.manual_only is True
    finally:
        store.close()


# --------------------------------------------------------------------------
# Phase 4.13.1 Fix #A — KILL_STATE_FORCE_RESET audit event wiring
# --------------------------------------------------------------------------


async def _run_daemon_self_check_with_corrupt_kill_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_reset: bool
) -> Path:
    """Pre-corrupt the kill DB, run daemon self-check, return audit DB path."""
    from executor.core.daemon import run_daemon

    monkeypatch.setenv("PAPER_MODE", "true")
    monkeypatch.setenv("EXECUTOR_PAPER_MODE_NO_ORDERBOOK", "true")
    if force_reset:
        monkeypatch.setenv(FORCE_RESET_ENV_VAR, "1")
    else:
        monkeypatch.delenv(FORCE_RESET_ENV_VAR, raising=False)

    kill_db = tmp_path / "kill.sqlite"
    kill_db.write_bytes(b"not a sqlite database (pre-corrupted for test)")

    rc = await run_daemon(
        self_check_only=True,
        audit_dir=tmp_path / "audit",
        risk_yaml="/root/executor/config/risk.yaml",
        risk_state_db=tmp_path / "rstate.sqlite",
        kill_db=kill_db,
        attribution_db=tmp_path / "attr.sqlite",
        telemetry_port=0,
        enable_quote_feeder=False,
    )
    # Self-check may fail because kill state seeded HARD+manual_only in the
    # non-force-reset path, but the audit DB will still have our corruption
    # events. The force-reset path seeds NONE so self-check should pass.
    assert rc in (0, 1)
    audit_files = list((tmp_path / "audit").glob("audit-*.sqlite"))
    assert len(audit_files) == 1
    return audit_files[0]


@pytest.mark.asyncio
async def test_force_reset_emits_audit_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With EXECUTOR_FORCE_RESET_KILL_STATE=1 + corrupt kill DB, both the
    existing ERROR(KILL_DB_REBUILT_FROM_CORRUPTION) and the new
    KILL_STATE_FORCE_RESET events must be in the audit log."""
    audit_db = await _run_daemon_self_check_with_corrupt_kill_db(
        tmp_path, monkeypatch, force_reset=True
    )
    conn = sqlite3.connect(str(audit_db))
    try:
        err_rows = conn.execute(
            "SELECT payload_json FROM events WHERE event_type='ERROR'"
        ).fetchall()
        fr_rows = conn.execute(
            "SELECT payload_json FROM events WHERE event_type='KILL_STATE_FORCE_RESET'"
        ).fetchall()
    finally:
        conn.close()

    # Exactly one KILL_STATE_FORCE_RESET event with correct payload.
    assert len(fr_rows) == 1, f"expected 1 KILL_STATE_FORCE_RESET, got {len(fr_rows)}"
    import json

    fr_payload = json.loads(fr_rows[0][0])
    assert "kill_db_path" in fr_payload
    assert fr_payload["ns_ts"] > 0
    assert "backup_path" in fr_payload
    assert "EXECUTOR_FORCE_RESET_KILL_STATE=1" in fr_payload["note"]

    # The rebuild ERROR event must also fire, and its note must reflect
    # the force-reset branch.
    corrupt_err = [
        json.loads(r[0])
        for r in err_rows
        if json.loads(r[0]).get("kind") == "KILL_DB_REBUILT_FROM_CORRUPTION"
    ]
    assert len(corrupt_err) == 1
    assert "force-reset" in corrupt_err[0]["note"]
    assert "mode=NONE" in corrupt_err[0]["note"]


@pytest.mark.asyncio
async def test_default_corruption_no_force_reset_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the bypass env var, only the ERROR event fires — no
    KILL_STATE_FORCE_RESET event in the audit log."""
    audit_db = await _run_daemon_self_check_with_corrupt_kill_db(
        tmp_path, monkeypatch, force_reset=False
    )
    conn = sqlite3.connect(str(audit_db))
    try:
        err_rows = conn.execute(
            "SELECT payload_json FROM events WHERE event_type='ERROR'"
        ).fetchall()
        fr_count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='KILL_STATE_FORCE_RESET'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert fr_count == 0, "KILL_STATE_FORCE_RESET must not fire without bypass"
    import json

    corrupt_err = [
        json.loads(r[0])
        for r in err_rows
        if json.loads(r[0]).get("kind") == "KILL_DB_REBUILT_FROM_CORRUPTION"
    ]
    assert len(corrupt_err) == 1
    # Note reflects fail-closed branch.
    assert "mode=HARD" in corrupt_err[0]["note"]
    assert "manual_only=True" in corrupt_err[0]["note"]


def test_corrupt_backup_filename_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two successive corruption cycles must yield two distinct backup
    files (ns-resolution timestamp suffix)."""
    monkeypatch.delenv(FORCE_RESET_ENV_VAR, raising=False)
    db_path = tmp_path / "kill_state.sqlite"

    db_path.write_bytes(b"corrupt round 1")
    store_a = KillStateStore(db_path)
    store_a.close()

    # The rebuild wrote a valid fresh DB with HARD seed. Clobber again.
    db_path.write_bytes(b"corrupt round 2 (different bytes)")
    store_b = KillStateStore(db_path)
    store_b.close()

    pattern = re.compile(r"^kill_state\.sqlite\.corrupt-(\d+)$")
    suffixes = sorted(
        int(m.group(1))
        for m in (pattern.match(p.name) for p in tmp_path.iterdir())
        if m
    )
    assert len(suffixes) == 2, f"expected 2 backups, got: {list(tmp_path.iterdir())}"
    assert suffixes[0] != suffixes[1]
