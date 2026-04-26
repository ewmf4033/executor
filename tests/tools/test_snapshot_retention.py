"""Tests for executor.tools.snapshot_retention — phase 5a.2.

All tests are hermetic: temp directories, no live rclone, no B2 access.
"""
from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Import-time safety: importing the module must not invoke subprocess.
# ---------------------------------------------------------------------------

def test_import_no_subprocess():
    """Importing snapshot_retention must not call subprocess at all."""
    with mock.patch("subprocess.run", side_effect=AssertionError("subprocess called at import")):
        with mock.patch("subprocess.Popen", side_effect=AssertionError("subprocess called at import")):
            # Force re-import.
            mod_name = "executor.tools.snapshot_retention"
            saved = sys.modules.pop(mod_name, None)
            try:
                import importlib
                importlib.import_module(mod_name)
            finally:
                if saved is not None:
                    sys.modules[mod_name] = saved

from executor.tools.snapshot_retention import (
    COMPRESS_CHUNK,
    DEFAULT_RESTORE_DIR,
    EXIT_LOCK_HELD,
    LOCK_STALE_SECONDS,
    RESTORED_SUFFIX,
    LockError,
    _acquire_lock,
    _compute_sha256,
    _discover,
    _is_owner_dead,
    _load_sidecar,
    _proc_start_time,
    _release_lock,
    _sidecar_path,
    _today_utc,
    _write_sidecar,
    build_parser,
    cmd_compress,
    cmd_inventory,
    cmd_prune,
    cmd_restore,
    cmd_upload,
    cmd_verify,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_JSONL = '{"ts":"2026-04-24T12:00:00Z","type":"trade","ticker":"FOO"}\n' * 10


def _make_snapshot(snap_dir: Path, date: str, content: str = SAMPLE_JSONL) -> Path:
    snap_dir.mkdir(parents=True, exist_ok=True)
    p = snap_dir / f"{date}.jsonl"
    p.write_text(content)
    return p


def _make_gz_and_sidecar(
    snap_dir: Path, date: str, content: str = SAMPLE_JSONL, *, verified: bool = False,
    remote_size_match: bool = True,
) -> tuple[Path, Path, Path]:
    """Create raw, gz, and sidecar. Returns (raw, gz, sidecar)."""
    raw = _make_snapshot(snap_dir, date, content)
    raw_size = raw.stat().st_size
    raw_sha256 = _compute_sha256(raw)

    gz_path = raw.with_suffix(raw.suffix + ".gz")
    with open(raw, "rb") as fin, gzip.open(gz_path, "wb") as fout:
        fout.write(fin.read())

    gz_stat = gz_path.stat()
    gz_sha256 = _compute_sha256(gz_path)

    meta: dict[str, Any] = {
        "date": date,
        "raw_name": raw.name,
        "raw_size": raw_size,
        "raw_sha256": raw_sha256,
        "gz_name": gz_path.name,
        "gz_size": gz_stat.st_size,
        "gz_sha256": gz_sha256,
        "compressed_at_utc": "2026-04-24T00:00:00+00:00",
        "verified_at_utc": "2026-04-24T01:00:00+00:00" if verified else None,
        "remote_size": gz_stat.st_size if (verified and remote_size_match) else None,
    }
    sidecar = _sidecar_path(gz_path)
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    return raw, gz_path, sidecar


def _ns(**kwargs: Any) -> Any:
    """Build a namespace that looks like parsed args."""
    from types import SimpleNamespace
    defaults = {"snapshot_dir": "/tmp", "verbose": False}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

class TestInventory:
    def test_empty_dir(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        snap.mkdir()
        rc = cmd_inventory(_ns(snapshot_dir=str(snap)))
        assert rc == 0

    def test_lists_files(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        snap = tmp_path / "snaps"
        _make_snapshot(snap, "2026-04-22")
        _make_snapshot(snap, "2026-04-23")
        rc = cmd_inventory(_ns(snapshot_dir=str(snap)))
        assert rc == 0
        out = capsys.readouterr().out
        assert "2026-04-22" in out
        assert "2026-04-23" in out

    def test_deterministic(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        snap = tmp_path / "snaps"
        _make_snapshot(snap, "2026-04-23")
        _make_snapshot(snap, "2026-04-22")
        _make_snapshot(snap, "2026-04-24")
        cmd_inventory(_ns(snapshot_dir=str(snap)))
        out1 = capsys.readouterr().out
        cmd_inventory(_ns(snapshot_dir=str(snap)))
        out2 = capsys.readouterr().out
        assert out1 == out2

    def test_flags_orphan_gz(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        snap = tmp_path / "snaps"
        snap.mkdir(parents=True, exist_ok=True)
        raw = _make_snapshot(snap, "2026-04-22")
        gz = raw.with_suffix(raw.suffix + ".gz")
        with open(raw, "rb") as fin, gzip.open(gz, "wb") as fout:
            fout.write(fin.read())
        # No sidecar → orphan.
        cmd_inventory(_ns(snapshot_dir=str(snap)))
        out = capsys.readouterr().out
        assert "ORPHAN-GZ" in out

    def test_flags_orphan_sidecar(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        snap = tmp_path / "snaps"
        snap.mkdir(parents=True, exist_ok=True)
        # Sidecar with no gz.
        sidecar = snap / "2026-04-22.jsonl.gz.meta.json"
        sidecar.write_text(json.dumps({"gz_size": 0}))
        cmd_inventory(_ns(snapshot_dir=str(snap)))
        out = capsys.readouterr().out
        assert "ORPHAN-SIDECAR" in out


# ---------------------------------------------------------------------------
# Today UTC / mtime guard
# ---------------------------------------------------------------------------

class TestTodayGuard:
    def test_today_not_compressed(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        snap = tmp_path / "snaps"
        today = _today_utc()
        raw = _make_snapshot(snap, today)
        # Set old mtime so only "today" logic protects it.
        old_time = time.time() - 86400
        os.utime(raw, (old_time, old_time))
        rc = cmd_compress(_ns(snapshot_dir=str(snap), execute=False))
        assert rc == 0
        out = capsys.readouterr().out
        assert "would compress" not in out

    def test_recent_mtime_not_compressed(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        snap = tmp_path / "snaps"
        raw = _make_snapshot(snap, "2020-01-01")
        # mtime = now → within grace.
        rc = cmd_compress(_ns(snapshot_dir=str(snap), execute=False))
        assert rc == 0
        out = capsys.readouterr().out
        assert "would compress" not in out


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

class TestCompress:
    def test_creates_gz_and_sidecar(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        raw = _make_snapshot(snap, "2026-04-20")
        old_time = time.time() - 86400
        os.utime(raw, (old_time, old_time))
        rc = cmd_compress(_ns(snapshot_dir=str(snap), execute=True))
        assert rc == 0
        gz = snap / "2026-04-20.jsonl.gz"
        sidecar = snap / "2026-04-20.jsonl.gz.meta.json"
        assert gz.exists()
        assert sidecar.exists()

    def test_raw_not_deleted(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        raw = _make_snapshot(snap, "2026-04-20")
        old_time = time.time() - 86400
        os.utime(raw, (old_time, old_time))
        cmd_compress(_ns(snapshot_dir=str(snap), execute=True))
        assert raw.exists()

    def test_gzip_roundtrip(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        content = "line1\nline2\nline3\n"
        raw = _make_snapshot(snap, "2026-04-20", content)
        old_time = time.time() - 86400
        os.utime(raw, (old_time, old_time))
        cmd_compress(_ns(snapshot_dir=str(snap), execute=True))
        gz = snap / "2026-04-20.jsonl.gz"
        with gzip.open(gz, "rt") as f:
            assert f.read() == content

    def test_sidecar_hashes_correct(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        content = "hello\n"
        raw = _make_snapshot(snap, "2026-04-20", content)
        old_time = time.time() - 86400
        os.utime(raw, (old_time, old_time))
        cmd_compress(_ns(snapshot_dir=str(snap), execute=True))
        sidecar = snap / "2026-04-20.jsonl.gz.meta.json"
        meta = json.loads(sidecar.read_text())
        gz = snap / "2026-04-20.jsonl.gz"
        assert meta["raw_size"] == len(content.encode())
        assert meta["raw_sha256"] == hashlib.sha256(content.encode()).hexdigest()
        assert meta["gz_size"] == gz.stat().st_size
        assert meta["gz_sha256"] == _compute_sha256(gz)


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------

class TestLock:
    def test_lock_prevents_concurrent(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        snap.mkdir()
        lock = _acquire_lock(snap)
        try:
            with pytest.raises(LockError):
                _acquire_lock(snap)
        finally:
            _release_lock(lock)

    def test_stale_lock_not_reclaimed_without_proof(self, tmp_path: Path):
        """If PID + start_time cannot prove owner dead, exit 10."""
        snap = tmp_path / "snaps"
        snap.mkdir()
        lock_path = snap / ".snapshot_retention.lock"
        # Write lock with our own PID (alive) but very old timestamp.
        lock_data = {
            "pid": os.getpid(),
            "start_time": _proc_start_time(os.getpid()),
            "acquired_at": time.time() - LOCK_STALE_SECONDS - 100,
        }
        lock_path.write_text(json.dumps(lock_data))
        # Lock is stale by age, but owner (us) is alive → exit 10.
        with pytest.raises(SystemExit) as exc_info:
            _acquire_lock(snap)
        assert exc_info.value.code == EXIT_LOCK_HELD

    def test_stale_lock_reclaimed_when_owner_dead(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        snap.mkdir()
        lock_path = snap / ".snapshot_retention.lock"
        # PID that doesn't exist.
        lock_data = {
            "pid": 2**22,  # Very unlikely to be alive.
            "start_time": 999999.0,
            "acquired_at": time.time() - LOCK_STALE_SECONDS - 100,
        }
        lock_path.write_text(json.dumps(lock_data))
        lock = _acquire_lock(snap)
        _release_lock(lock)


# ---------------------------------------------------------------------------
# Upload dry-run
# ---------------------------------------------------------------------------

class TestUploadDryRun:
    def test_zero_subprocess_calls(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        _make_gz_and_sidecar(snap, "2026-04-22")
        with mock.patch("subprocess.run", side_effect=AssertionError("subprocess called")) as m:
            with mock.patch("subprocess.Popen", side_effect=AssertionError("subprocess called")):
                rc = cmd_upload(_ns(
                    snapshot_dir=str(snap),
                    i_confirm_snapshot_upload=False,
                    verbose=False,
                ))
        assert rc == 0
        m.assert_not_called()

    def test_dry_run_skips_orphan_gz(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        snap = tmp_path / "snaps"
        snap.mkdir(parents=True)
        raw = _make_snapshot(snap, "2026-04-22")
        gz = raw.with_suffix(raw.suffix + ".gz")
        with open(raw, "rb") as fin, gzip.open(gz, "wb") as fout:
            fout.write(fin.read())
        # No sidecar.
        with mock.patch("subprocess.run", side_effect=AssertionError("subprocess called")):
            rc = cmd_upload(_ns(
                snapshot_dir=str(snap),
                i_confirm_snapshot_upload=False,
                verbose=False,
            ))
        assert rc == 0
        out = capsys.readouterr().out
        assert "orphan gz" in out.lower()


# ---------------------------------------------------------------------------
# Upload live preflight
# ---------------------------------------------------------------------------

class TestUploadLivePreflight:
    def test_recomputes_hash_before_upload(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        _, gz, sidecar = _make_gz_and_sidecar(snap, "2026-04-22")
        call_log: list[str] = []

        def fake_run(cmd, **kw):
            call_log.append(cmd[1] if len(cmd) > 1 else cmd[0])
            r = mock.Mock()
            r.returncode = 0
            r.stdout = ""
            r.stderr = ""
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            rc = cmd_upload(_ns(
                snapshot_dir=str(snap),
                i_confirm_snapshot_upload=True,
                verbose=False,
            ))
        assert rc == 0
        assert "copyto" in call_log

    def test_refuses_gz_sidecar_mismatch(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        _, gz, sidecar = _make_gz_and_sidecar(snap, "2026-04-22")
        # Corrupt sidecar.
        meta = json.loads(sidecar.read_text())
        meta["gz_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        sidecar.write_text(json.dumps(meta))
        with mock.patch("subprocess.run", side_effect=AssertionError("should not call")):
            rc = cmd_upload(_ns(
                snapshot_dir=str(snap),
                i_confirm_snapshot_upload=True,
                verbose=False,
            ))
        assert rc == 1


# ---------------------------------------------------------------------------
# Orphan policy
# ---------------------------------------------------------------------------

class TestOrphanPolicy:
    def test_orphan_gz_not_uploadable(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        snap = tmp_path / "snaps"
        snap.mkdir(parents=True)
        raw = _make_snapshot(snap, "2026-04-22")
        gz = raw.with_suffix(raw.suffix + ".gz")
        with open(raw, "rb") as fin, gzip.open(gz, "wb") as fout:
            fout.write(fin.read())
        # No sidecar → orphan.
        with mock.patch("subprocess.run", side_effect=AssertionError("subprocess called")):
            rc = cmd_upload(_ns(
                snapshot_dir=str(snap),
                i_confirm_snapshot_upload=False,
                verbose=False,
            ))
        assert rc == 0
        assert "orphan" in capsys.readouterr().out.lower()

    def test_orphan_gz_not_prunable(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        snap = tmp_path / "snaps"
        snap.mkdir(parents=True)
        raw = _make_snapshot(snap, "2026-04-22")
        gz = raw.with_suffix(raw.suffix + ".gz")
        with open(raw, "rb") as fin, gzip.open(gz, "wb") as fout:
            fout.write(fin.read())
        rc = cmd_prune(_ns(snapshot_dir=str(snap), execute=True))
        assert rc == 0
        assert gz.exists()  # Not pruned.

    def test_orphan_sidecar_flagged_inventory(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        snap = tmp_path / "snaps"
        snap.mkdir(parents=True)
        sidecar = snap / "2026-04-22.jsonl.gz.meta.json"
        sidecar.write_text(json.dumps({"gz_size": 0}))
        cmd_inventory(_ns(snapshot_dir=str(snap)))
        out = capsys.readouterr().out
        assert "ORPHAN-SIDECAR" in out

    def test_orphan_sidecar_not_uploadable_prunable(self, tmp_path: Path):
        """Orphan sidecar (no .gz) must not cause upload or prune to process it."""
        snap = tmp_path / "snaps"
        snap.mkdir(parents=True)
        sidecar = snap / "2026-04-22.jsonl.gz.meta.json"
        sidecar.write_text(json.dumps({"gz_size": 0}))
        # Upload dry-run should not crash.
        rc_upload = cmd_upload(_ns(
            snapshot_dir=str(snap),
            i_confirm_snapshot_upload=False,
            verbose=False,
        ))
        assert rc_upload == 0
        # Prune should not crash.
        rc_prune = cmd_prune(_ns(snapshot_dir=str(snap), execute=False))
        assert rc_prune == 0


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

class TestVerify:
    def test_requires_remote_size_match(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        _, gz, sidecar = _make_gz_and_sidecar(snap, "2026-04-22")
        meta = json.loads(sidecar.read_text())
        wrong_size = meta["gz_size"] + 100

        def fake_run(cmd, **kw):
            r = mock.Mock()
            r.returncode = 0
            r.stdout = json.dumps({"bytes": wrong_size, "count": 1})
            r.stderr = ""
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            rc = cmd_verify(_ns(snapshot_dir=str(snap), date="2026-04-22"))
        assert rc == 1

    def test_verify_updates_sidecar(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        _, gz, sidecar = _make_gz_and_sidecar(snap, "2026-04-22")
        meta = json.loads(sidecar.read_text())
        correct_size = meta["gz_size"]

        def fake_run(cmd, **kw):
            r = mock.Mock()
            r.returncode = 0
            r.stdout = json.dumps({"bytes": correct_size, "count": 1})
            r.stderr = ""
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            rc = cmd_verify(_ns(snapshot_dir=str(snap), date="2026-04-22"))
        assert rc == 0
        updated = json.loads(sidecar.read_text())
        assert updated["verified_at_utc"] is not None
        assert updated["remote_size"] == correct_size


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------

class TestPrune:
    def test_freshly_reads_sidecar(self, tmp_path: Path):
        """Prune must reload sidecar from disk, not use stale data."""
        snap = tmp_path / "snaps"
        _, gz, sidecar = _make_gz_and_sidecar(snap, "2026-04-22", verified=True)
        # Tamper sidecar after creation to clear verified_at_utc.
        meta = json.loads(sidecar.read_text())
        meta["verified_at_utc"] = None
        sidecar.write_text(json.dumps(meta))
        rc = cmd_prune(_ns(snapshot_dir=str(snap), execute=True))
        assert rc == 0
        assert gz.exists()  # Not pruned because sidecar freshly says unverified.

    def test_refuses_unverified(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        _, gz, _ = _make_gz_and_sidecar(snap, "2026-04-22", verified=False)
        rc = cmd_prune(_ns(snapshot_dir=str(snap), execute=True))
        assert rc == 0
        assert gz.exists()

    def test_refuses_remote_size_mismatch(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        _, gz, sidecar = _make_gz_and_sidecar(snap, "2026-04-22", verified=True)
        meta = json.loads(sidecar.read_text())
        meta["remote_size"] = meta["gz_size"] + 1  # Mismatch.
        sidecar.write_text(json.dumps(meta))
        rc = cmd_prune(_ns(snapshot_dir=str(snap), execute=True))
        assert rc == 0
        assert gz.exists()

    def test_refuses_local_gz_hash_mismatch(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        _, gz, sidecar = _make_gz_and_sidecar(snap, "2026-04-22", verified=True)
        # Corrupt the gz file.
        with open(gz, "ab") as f:
            f.write(b"CORRUPT")
        rc = cmd_prune(_ns(snapshot_dir=str(snap), execute=True))
        assert rc == 0
        assert gz.exists()

    def test_prune_succeeds_when_verified(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        _, gz, sidecar = _make_gz_and_sidecar(snap, "2026-04-22", verified=True)
        rc = cmd_prune(_ns(snapshot_dir=str(snap), execute=True))
        assert rc == 0
        assert not gz.exists()


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

class TestRestore:
    def test_refuses_overwrite_without_force(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        snap.mkdir()
        dest = tmp_path / "restore"
        dest.mkdir()
        existing = dest / ("2026-04-22.jsonl.gz" + RESTORED_SUFFIX)
        existing.write_text("old")
        rc = cmd_restore(_ns(
            snapshot_dir=str(snap), date="2026-04-22",
            dest=str(dest), force=False,
            allow_snapshot_dir_restore=False,
        ))
        assert rc == 1

    def test_default_dest_outside_snapshot_dir(self):
        assert str(DEFAULT_RESTORE_DIR) != str(
            Path(__file__).resolve().parents[2] / "audit-logs" / "market_snapshots" / "kalshi_ws"
        )

    def test_inside_snapshot_dir_requires_flag(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        snap.mkdir()
        rc = cmd_restore(_ns(
            snapshot_dir=str(snap), date="2026-04-22",
            dest=str(snap), force=False,
            allow_snapshot_dir_restore=False,
        ))
        assert rc == 1

    def test_inside_snapshot_dir_allowed_with_flag(self, tmp_path: Path):
        snap = tmp_path / "snaps"
        snap.mkdir()
        content = b"test content"
        gz_data = gzip.compress(content)

        def fake_run(cmd, **kw):
            # Simulate rclone copyto by writing gz file.
            dest_path = Path(cmd[-1])
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(gz_data)
            r = mock.Mock()
            r.returncode = 0
            r.stderr = ""
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            rc = cmd_restore(_ns(
                snapshot_dir=str(snap), date="2026-04-22",
                dest=str(snap), force=False,
                allow_snapshot_dir_restore=True,
            ))
        assert rc == 0


# ---------------------------------------------------------------------------
# Subprocess stderr hygiene
# ---------------------------------------------------------------------------

class TestStderrHygiene:
    def test_stderr_not_printed_by_default(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        snap = tmp_path / "snaps"
        _, gz, sidecar = _make_gz_and_sidecar(snap, "2026-04-22")

        def fake_run(cmd, **kw):
            r = mock.Mock()
            r.returncode = 1
            r.stdout = ""
            r.stderr = "SECRET_INTERNAL_ERROR_DETAILS"
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            rc = cmd_upload(_ns(
                snapshot_dir=str(snap),
                i_confirm_snapshot_upload=True,
                verbose=False,
            ))
        assert rc == 1
        captured = capsys.readouterr()
        assert "SECRET_INTERNAL_ERROR_DETAILS" not in captured.out
        assert "SECRET_INTERNAL_ERROR_DETAILS" not in captured.err

    def test_stderr_shown_in_verbose(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        snap = tmp_path / "snaps"
        _, gz, sidecar = _make_gz_and_sidecar(snap, "2026-04-22")

        def fake_run(cmd, **kw):
            r = mock.Mock()
            r.returncode = 1
            r.stdout = ""
            r.stderr = "VERBOSE_ERROR_DETAILS"
            return r

        with mock.patch("subprocess.run", side_effect=fake_run):
            rc = cmd_upload(_ns(
                snapshot_dir=str(snap),
                i_confirm_snapshot_upload=True,
                verbose=True,
            ))
        assert rc == 1
        captured = capsys.readouterr()
        assert "VERBOSE_ERROR_DETAILS" in captured.err
