"""Persistence round-trip for KillStateStore."""
from __future__ import annotations

import re
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
