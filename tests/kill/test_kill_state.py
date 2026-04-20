"""Persistence round-trip for KillStateStore."""
from __future__ import annotations

from pathlib import Path

from executor.kill.state import KillMode, KillStateSnapshot, KillStateStore


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
