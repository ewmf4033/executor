"""Phase 5a.2 — snapshot retention / B2 upload for Kalshi WS JSONL files.

Manual-only, subcommand-driven tool for managing lifecycle of Kalshi
WebSocket snapshot files stored at::

    audit-logs/market_snapshots/kalshi_ws/YYYY-MM-DD.jsonl

Subcommands:
    inventory   List snapshots with status (raw / compressed / uploaded / orphan).
    compress    Gzip eligible .jsonl files, write sidecar metadata.
    upload      Upload .jsonl.gz to B2 via rclone.
    verify      Verify remote copy matches local sidecar metadata.
    prune       Delete local .jsonl.gz after verified upload.
    restore     Download .jsonl.gz from B2 and decompress.

All operations default to dry-run.  Pass ``--execute`` (compress/prune) or
``--i-confirm-snapshot-upload`` (upload) to perform real work.

Module-level imports are stdlib-only.  No subprocess or rclone interaction
occurs at import time.

Run as a module::

    python3 -m executor.tools.snapshot_retention inventory
    python3 -m executor.tools.snapshot_retention compress --execute
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import gzip
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "audit-logs" / "market_snapshots" / "kalshi_ws"
DEFAULT_RESTORE_DIR = REPO_ROOT / "restored_snapshots"

RCLONE_REMOTE = "b2backup"
B2_BUCKET = "ari-executor-backups"
B2_PREFIX = "kalshi_ws_snapshots/"

RAW_RETENTION_DAYS = 2
COMPRESSED_RETENTION_DAYS = 14

MTIME_GRACE_SECONDS = 15 * 60  # 15 minutes
MIN_FREE_DISK_MB = 1500

LOCK_STALE_SECONDS = 3600  # 1 hour

JSONL_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")
GZ_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl\.gz$")
SIDECAR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl\.gz\.meta\.json$")

COMPRESS_CHUNK = 256 * 1024  # 256 KiB streaming chunks

EXIT_LOCK_HELD = 10

# Restored file suffix so prune ignores them.
RESTORED_SUFFIX = ".restored"


# ---------------------------------------------------------------------------
# Lockfile
# ---------------------------------------------------------------------------

class LockError(Exception):
    """Raised when the lock cannot be acquired."""


def _read_lock_info(lock_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(lock_path.read_text())
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _proc_start_time(pid: int) -> float | None:
    """Read process start time from /proc/<pid>/stat (Linux).

    Returns clock ticks since boot for the start time field, or None if
    the process does not exist or /proc is unavailable.
    """
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text()
        # Fields after the comm (which may contain spaces/parens).
        # comm is enclosed in parens; find the last ')'.
        close_paren = stat_line.rfind(")")
        fields = stat_line[close_paren + 2:].split()
        # Field index 0 after comm = state, index 19 = starttime (22nd overall).
        return float(fields[19])
    except (FileNotFoundError, OSError, IndexError, ValueError):
        return None


def _is_owner_dead(lock_info: dict[str, Any]) -> bool:
    """Return True only if both PID and start-time prove the owner is dead."""
    pid = lock_info.get("pid")
    recorded_start = lock_info.get("start_time")
    if pid is None or recorded_start is None:
        return False  # Cannot prove dead → refuse.

    current_start = _proc_start_time(pid)
    if current_start is None:
        # Process does not exist at all → dead.
        return True
    # PID exists but start time differs → PID was recycled.
    return current_start != recorded_start


def _acquire_lock(snapshot_dir: Path) -> Path:
    """Acquire a lockfile.  Returns the lock path on success.

    Raises LockError if the lock is held by a live process.
    Exits with EXIT_LOCK_HELD if the lock is stale but owner liveness
    cannot be determined.
    """
    lock_path = snapshot_dir / ".snapshot_retention.lock"
    now = time.time()

    existing = _read_lock_info(lock_path)
    if existing is not None:
        lock_age = now - existing.get("acquired_at", now)
        if lock_age < LOCK_STALE_SECONDS:
            raise LockError(
                f"Lock held by PID {existing.get('pid')} "
                f"(age {lock_age:.0f}s < stale threshold {LOCK_STALE_SECONDS}s)"
            )
        # Stale by age — but must prove owner dead via PID + start time.
        if not _is_owner_dead(existing):
            sys.exit(EXIT_LOCK_HELD)
        # Owner provably dead — reclaim.

    my_pid = os.getpid()
    my_start = _proc_start_time(my_pid)
    lock_data = {
        "pid": my_pid,
        "start_time": my_start,
        "acquired_at": now,
    }
    lock_path.write_text(json.dumps(lock_data))
    return lock_path


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Sidecar helpers
# ---------------------------------------------------------------------------

def _sidecar_path(gz_path: Path) -> Path:
    return gz_path.with_suffix(gz_path.suffix + ".meta.json")


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(COMPRESS_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_sidecar(gz_path: Path, raw_path: Path, raw_size: int, raw_sha256: str) -> Path:
    gz_stat = gz_path.stat()
    gz_sha256 = _compute_sha256(gz_path)
    sidecar = _sidecar_path(gz_path)
    meta: dict[str, Any] = {
        "date": gz_path.stem.replace(".jsonl", ""),
        "raw_name": raw_path.name,
        "raw_size": raw_size,
        "raw_sha256": raw_sha256,
        "gz_name": gz_path.name,
        "gz_size": gz_stat.st_size,
        "gz_sha256": gz_sha256,
        "compressed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verified_at_utc": None,
        "remote_size": None,
    }
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    return sidecar


def _load_sidecar(sidecar: Path) -> dict[str, Any] | None:
    try:
        return json.loads(sidecar.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _today_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _discover(snapshot_dir: Path) -> dict[str, dict[str, Any]]:
    """Discover snapshot artifacts grouped by date.

    Returns ``{date_str: {"raw": Path|None, "gz": Path|None, "sidecar": Path|None}}``.
    """
    dates: dict[str, dict[str, Any]] = {}

    if not snapshot_dir.is_dir():
        return dates

    for entry in sorted(snapshot_dir.iterdir()):
        name = entry.name
        date_str: str | None = None
        kind: str | None = None

        if SIDECAR_PATTERN.match(name):
            date_str = name[:10]
            kind = "sidecar"
        elif GZ_PATTERN.match(name):
            date_str = name[:10]
            kind = "gz"
        elif JSONL_PATTERN.match(name):
            date_str = name[:10]
            kind = "raw"
        else:
            continue

        if date_str not in dates:
            dates[date_str] = {"raw": None, "gz": None, "sidecar": None}
        dates[date_str][kind] = entry

    return dates


# ---------------------------------------------------------------------------
# Disk space check
# ---------------------------------------------------------------------------

def _free_disk_mb(path: Path) -> float:
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_inventory(args: argparse.Namespace) -> int:
    snapshot_dir = Path(args.snapshot_dir)
    dates = _discover(snapshot_dir)
    today = _today_utc()

    if not dates:
        print("No snapshot files found.")
        return 0

    print(f"{'Date':<12} {'Raw':>10} {'GZ':>10} {'Sidecar':>8} {'Status'}")
    print("-" * 60)

    for date_str in sorted(dates):
        info = dates[date_str]
        raw_size = info["raw"].stat().st_size if info["raw"] else 0
        gz_size = info["gz"].stat().st_size if info["gz"] else 0
        has_sidecar = info["sidecar"] is not None

        # Determine status.
        flags: list[str] = []
        if date_str == today:
            flags.append("active")
        if info["raw"] and not info["gz"]:
            flags.append("raw-only")
        if info["gz"] and not info["sidecar"]:
            flags.append("ORPHAN-GZ")
        if info["sidecar"] and not info["gz"]:
            flags.append("ORPHAN-SIDECAR")
        if info["gz"] and info["sidecar"]:
            meta = _load_sidecar(info["sidecar"])
            if meta and meta.get("verified_at_utc"):
                flags.append("verified")
            elif meta:
                flags.append("compressed")

        print(
            f"{date_str:<12} "
            f"{raw_size:>10,} "
            f"{gz_size:>10,} "
            f"{'Y' if has_sidecar else 'N':>8} "
            f"{', '.join(flags)}"
        )

    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    snapshot_dir = Path(args.snapshot_dir)
    execute = args.execute
    dates = _discover(snapshot_dir)
    today = _today_utc()
    now = time.time()

    lock_path: Path | None = None
    if execute:
        lock_path = _acquire_lock(snapshot_dir)

    try:
        compressed = 0
        skipped = 0

        for date_str in sorted(dates):
            info = dates[date_str]
            raw = info["raw"]
            if raw is None:
                continue
            if info["gz"] is not None:
                continue  # Already compressed.
            if date_str == today:
                skipped += 1
                continue
            # Skip recent mtime.
            try:
                mtime = raw.stat().st_mtime
            except OSError:
                continue
            if (now - mtime) < MTIME_GRACE_SECONDS:
                skipped += 1
                continue

            if not execute:
                print(f"[dry-run] would compress {raw.name}")
                compressed += 1
                continue

            # Check disk space.
            if _free_disk_mb(snapshot_dir) < MIN_FREE_DISK_MB:
                print(f"ERROR: free disk below {MIN_FREE_DISK_MB} MB, stopping.", file=sys.stderr)
                return 1

            # Pre-compress stat + hash.
            pre_stat = raw.stat()
            raw_size = pre_stat.st_size
            raw_sha256 = _compute_sha256(raw)

            gz_path = raw.with_suffix(raw.suffix + ".gz")
            tmp_path = gz_path.with_suffix(".gz.tmp")

            # Streaming compress.
            with open(raw, "rb") as fin, gzip.open(tmp_path, "wb") as fout:
                while True:
                    chunk = fin.read(COMPRESS_CHUNK)
                    if not chunk:
                        break
                    fout.write(chunk)

            # fsync + atomic rename.
            fd = os.open(str(tmp_path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            tmp_path.rename(gz_path)

            # Post-compress consistency: raw file unchanged.
            post_stat = raw.stat()
            if post_stat.st_size != raw_size or post_stat.st_mtime != pre_stat.st_mtime:
                print(f"WARNING: {raw.name} changed during compression", file=sys.stderr)

            # Write sidecar.
            _write_sidecar(gz_path, raw, raw_size, raw_sha256)
            print(f"compressed {raw.name} -> {gz_path.name}")
            compressed += 1

        print(f"\nCompressed: {compressed}, Skipped: {skipped}")
        return 0
    finally:
        if lock_path:
            _release_lock(lock_path)


def cmd_upload(args: argparse.Namespace) -> int:
    import subprocess  # Lazy import — never at module level.

    snapshot_dir = Path(args.snapshot_dir)
    dry_run = not args.i_confirm_snapshot_upload
    verbose = args.verbose
    dates = _discover(snapshot_dir)

    if dry_run:
        # Dry-run: ZERO subprocess calls.
        for date_str in sorted(dates):
            info = dates[date_str]
            gz = info["gz"]
            sidecar = info["sidecar"]
            if gz is None:
                continue
            if sidecar is None:
                print(f"[dry-run] SKIP {gz.name}: orphan gz (no sidecar)")
                continue
            meta = _load_sidecar(sidecar)
            if meta is None:
                print(f"[dry-run] SKIP {gz.name}: unreadable sidecar")
                continue
            remote_dest = f"{RCLONE_REMOTE}:{B2_BUCKET}/{B2_PREFIX}{gz.name}"
            print(f"[dry-run] would upload {gz.name} -> {remote_dest}")
        return 0

    # Live upload path.
    lock_path = _acquire_lock(snapshot_dir)
    try:
        uploaded = 0
        for date_str in sorted(dates):
            info = dates[date_str]
            gz = info["gz"]
            sidecar = info["sidecar"]
            if gz is None:
                continue
            if sidecar is None:
                print(f"SKIP {gz.name}: orphan gz (no sidecar)", file=sys.stderr)
                continue
            meta = _load_sidecar(sidecar)
            if meta is None:
                print(f"SKIP {gz.name}: unreadable sidecar", file=sys.stderr)
                continue

            # Upload preflight integrity (Codex constraint #2).
            current_size = gz.stat().st_size
            current_sha256 = _compute_sha256(gz)
            if current_size != meta["gz_size"] or current_sha256 != meta["gz_sha256"]:
                print(
                    f"ERROR: {gz.name} integrity mismatch — "
                    f"expected size={meta['gz_size']} sha256={meta['gz_sha256']}, "
                    f"got size={current_size} sha256={current_sha256}",
                    file=sys.stderr,
                )
                return 1

            remote_dest = f"{RCLONE_REMOTE}:{B2_BUCKET}/{B2_PREFIX}{gz.name}"
            cmd = ["rclone", "copyto", str(gz), remote_dest]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                msg = "rclone upload failed"
                if verbose:
                    msg += f": {result.stderr.strip()}"
                print(f"ERROR: {msg}", file=sys.stderr)
                return 1
            print(f"uploaded {gz.name} -> {remote_dest}")
            uploaded += 1

        print(f"\nUploaded: {uploaded}")
        return 0
    finally:
        _release_lock(lock_path)


def cmd_verify(args: argparse.Namespace) -> int:
    import subprocess

    snapshot_dir = Path(args.snapshot_dir)
    date_str = args.date
    verbose = args.verbose

    gz_path = snapshot_dir / f"{date_str}.jsonl.gz"
    sidecar = _sidecar_path(gz_path)

    meta = _load_sidecar(sidecar)
    if meta is None:
        print(f"ERROR: no sidecar for {date_str}", file=sys.stderr)
        return 1

    remote_path = f"{RCLONE_REMOTE}:{B2_BUCKET}/{B2_PREFIX}{gz_path.name}"
    cmd = ["rclone", "size", "--json", remote_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = "rclone size failed"
        if verbose:
            msg += f": {result.stderr.strip()}"
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    try:
        size_info = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("ERROR: could not parse rclone size output", file=sys.stderr)
        return 1

    remote_size = size_info.get("bytes", -1)
    expected_size = meta["gz_size"]

    if remote_size != expected_size:
        print(
            f"VERIFY FAIL: remote size {remote_size} != expected {expected_size}",
            file=sys.stderr,
        )
        return 1

    # Update sidecar with verification timestamp and remote size.
    meta["verified_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["remote_size"] = remote_size
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"verified {date_str}: remote_size={remote_size} matches gz_size={expected_size}")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    snapshot_dir = Path(args.snapshot_dir)
    execute = args.execute
    dates = _discover(snapshot_dir)

    lock_path: Path | None = None
    if execute:
        lock_path = _acquire_lock(snapshot_dir)

    try:
        pruned = 0
        refused = 0

        for date_str in sorted(dates):
            info = dates[date_str]
            gz = info["gz"]
            if gz is None:
                continue
            sidecar_p = info["sidecar"]
            if sidecar_p is None:
                # Orphan gz — not prunable (Codex #3).
                refused += 1
                continue

            # Freshly load sidecar (Codex #6 — never trust cached inventory).
            meta = _load_sidecar(sidecar_p)
            if meta is None:
                refused += 1
                continue

            # Gate: verified_at_utc must be set.
            if not meta.get("verified_at_utc"):
                if execute:
                    print(f"REFUSE prune {gz.name}: not verified", file=sys.stderr)
                refused += 1
                continue

            # Gate: remote_size must equal gz_size.
            if meta.get("remote_size") != meta.get("gz_size"):
                if execute:
                    print(f"REFUSE prune {gz.name}: remote_size mismatch", file=sys.stderr)
                refused += 1
                continue

            # Gate: local gz integrity.
            try:
                current_size = gz.stat().st_size
                current_sha256 = _compute_sha256(gz)
            except OSError:
                refused += 1
                continue

            if current_size != meta["gz_size"] or current_sha256 != meta["gz_sha256"]:
                if execute:
                    print(f"REFUSE prune {gz.name}: local integrity mismatch", file=sys.stderr)
                refused += 1
                continue

            if not execute:
                print(f"[dry-run] would prune {gz.name}")
                pruned += 1
                continue

            gz.unlink()
            # Also remove raw if present.
            raw = info["raw"]
            if raw and raw.exists():
                raw.unlink()
            print(f"pruned {gz.name}")
            pruned += 1

        print(f"\nPruned: {pruned}, Refused: {refused}")
        return 0
    finally:
        if lock_path:
            _release_lock(lock_path)


def cmd_restore(args: argparse.Namespace) -> int:
    import subprocess

    snapshot_dir = Path(args.snapshot_dir)
    date_str = args.date
    dest = Path(args.dest) if args.dest else DEFAULT_RESTORE_DIR
    force = args.force
    allow_snap = args.allow_snapshot_dir_restore
    verbose = args.verbose

    # Safety: refuse restore inside snapshot_dir without explicit flag.
    try:
        dest_resolved = dest.resolve()
        snap_resolved = snapshot_dir.resolve()
        if dest_resolved == snap_resolved or str(dest_resolved).startswith(str(snap_resolved) + os.sep):
            if not allow_snap:
                print(
                    "ERROR: --dest is inside snapshot-dir. "
                    "Pass --allow-snapshot-dir-restore to proceed.",
                    file=sys.stderr,
                )
                return 1
    except OSError:
        pass

    dest.mkdir(parents=True, exist_ok=True)

    gz_name = f"{date_str}.jsonl.gz"
    local_gz = dest / (gz_name + RESTORED_SUFFIX)
    local_jsonl = dest / (f"{date_str}.jsonl" + RESTORED_SUFFIX)

    if local_gz.exists() and not force:
        print(f"ERROR: {local_gz} exists. Use --force to overwrite.", file=sys.stderr)
        return 1
    if local_jsonl.exists() and not force:
        print(f"ERROR: {local_jsonl} exists. Use --force to overwrite.", file=sys.stderr)
        return 1

    remote_path = f"{RCLONE_REMOTE}:{B2_BUCKET}/{B2_PREFIX}{gz_name}"
    cmd = ["rclone", "copyto", remote_path, str(local_gz)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = "rclone restore failed"
        if verbose:
            msg += f": {result.stderr.strip()}"
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    # Decompress.
    with gzip.open(local_gz, "rb") as fin, open(local_jsonl, "wb") as fout:
        while True:
            chunk = fin.read(COMPRESS_CHUNK)
            if not chunk:
                break
            fout.write(chunk)

    print(f"restored {date_str} -> {local_jsonl}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snapshot_retention",
        description="Kalshi WS snapshot retention manager",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=str(DEFAULT_SNAPSHOT_DIR),
        help="Path to snapshot directory (default: repo-relative).",
    )
    parser.add_argument("--verbose", action="store_true", default=False)

    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("inventory", help="List snapshot files and status.")

    p_compress = subs.add_parser("compress", help="Compress eligible .jsonl files.")
    p_compress.add_argument("--execute", action="store_true", default=False)

    p_upload = subs.add_parser("upload", help="Upload .jsonl.gz to B2.")
    p_upload.add_argument("--dry-run", action="store_true", default=True)
    p_upload.add_argument("--i-confirm-snapshot-upload", action="store_true", default=False)

    p_verify = subs.add_parser("verify", help="Verify remote copy.")
    p_verify.add_argument("--date", required=True, help="YYYY-MM-DD")

    p_prune = subs.add_parser("prune", help="Prune verified local gzip files.")
    p_prune.add_argument("--execute", action="store_true", default=False)

    p_restore = subs.add_parser("restore", help="Restore from B2.")
    p_restore.add_argument("--date", required=True, help="YYYY-MM-DD")
    p_restore.add_argument("--dest", default=None)
    p_restore.add_argument("--force", action="store_true", default=False)
    p_restore.add_argument("--allow-snapshot-dir-restore", action="store_true", default=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "inventory": cmd_inventory,
        "compress": cmd_compress,
        "upload": cmd_upload,
        "verify": cmd_verify,
        "prune": cmd_prune,
        "restore": cmd_restore,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
