"""
Phase 4.16 — point-sample host-health probes (stdlib only).

No background tasks, no external dependencies (no psutil). Each probe is a
synchronous, side-effect-free helper invoked from HostHealthGate.check().
On any unexpected error, probes raise; the gate decides whether to
fail-closed (capital mode) or approve with metadata (paper bypass).
"""
from __future__ import annotations

import os
import resource
import shutil
from typing import Any


# Default Linux paths. Tests override via the `_path` arguments.
DISK_PATH = "/"
MEMINFO_PATH = "/proc/meminfo"


def disk_pct(path: str = DISK_PATH) -> float:
    """Return percent of disk used at `path` in [0.0, 100.0]."""
    u = shutil.disk_usage(path)
    if u.total <= 0:
        return 0.0
    return (u.used / u.total) * 100.0


def inode_pct(path: str = DISK_PATH) -> float:
    """Return percent of inodes used at `path` in [0.0, 100.0].

    Uses os.statvfs.f_files (total) and f_favail (free for unprivileged).
    """
    s = os.statvfs(path)
    if s.f_files <= 0:
        return 0.0
    used = s.f_files - s.f_favail
    return (used / s.f_files) * 100.0


def swap_pct(meminfo_path: str = MEMINFO_PATH) -> float:
    """Parse /proc/meminfo and return swap-used percent in [0.0, 100.0].

    Returns 0.0 when SwapTotal is 0 (swap disabled). Raises RuntimeError
    if SwapTotal/SwapFree fields are absent.
    """
    with open(meminfo_path, "r", encoding="utf-8") as f:
        content = f.read()
    total_kb: int | None = None
    free_kb: int | None = None
    for line in content.splitlines():
        if line.startswith("SwapTotal:"):
            total_kb = int(line.split()[1])
        elif line.startswith("SwapFree:"):
            free_kb = int(line.split()[1])
        if total_kb is not None and free_kb is not None:
            break
    if total_kb is None or free_kb is None:
        raise RuntimeError(
            f"swap fields not found in {meminfo_path} "
            f"(SwapTotal={total_kb}, SwapFree={free_kb})"
        )
    if total_kb <= 0:
        return 0.0
    used_kb = total_kb - free_kb
    return (used_kb / total_kb) * 100.0


def rss_mb() -> float:
    """Return current process resident set size in megabytes.

    Linux: getrusage(RUSAGE_SELF).ru_maxrss is in kilobytes.
    """
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_maxrss / 1024.0


def loadavg_1m() -> float:
    """Return 1-minute load average (Linux-only)."""
    return os.getloadavg()[0]


def sample_host(
    *,
    check_rss: bool = False,
    check_loadavg: bool = False,
    disk_path: str = DISK_PATH,
    meminfo_path: str = MEMINFO_PATH,
) -> dict[str, Any]:
    """Point-sample all host-health probes.

    Optional probes (RSS, loadavg) are evaluated only when requested by
    the caller — typically the gate sets these based on whether the
    corresponding threshold is configured non-zero.

    Raises on any underlying probe error so the gate can decide between
    fail-closed (capital mode) and approve-with-metadata (paper bypass).
    """
    out: dict[str, Any] = {
        "disk_pct": round(disk_pct(disk_path), 2),
        "inode_pct": round(inode_pct(disk_path), 2),
        "swap_pct": round(swap_pct(meminfo_path), 2),
    }
    if check_rss:
        out["rss_mb"] = round(rss_mb(), 2)
    if check_loadavg:
        out["loadavg_1m"] = round(loadavg_1m(), 2)
    return out
