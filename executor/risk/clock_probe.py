"""
Phase 4.16 — point-sample clock-health probes (stdlib only).

No background task, no external NTP client library. Detects:
  - wall-clock regression (wall delta < 0 since last probe)
  - monotonic vs wall-clock skew (|wall_delta - mono_delta| > threshold)
  - NTP synchronization via `timedatectl show -p NTPSynchronized --value`

Module-level baseline state is reset on daemon restart; this is acceptable
for 4.16 (4.16 implements detection only, not historical drift tracking).
"""
from __future__ import annotations

import subprocess
import time
from typing import Any


# Indirection so tests can monkeypatch without touching `time` itself.
_now_wall_ns = time.time_ns
_now_monotonic_ns = time.monotonic_ns
_subprocess_run = subprocess.run

# Baseline state — module-level, reset on daemon restart by design.
_last_wall_ns: int | None = None
_last_monotonic_ns: int | None = None


def reset_baseline() -> None:
    """Clear baseline so the next sample_clock() returns
    status='first_probe_no_baseline'. Exposed for tests."""
    global _last_wall_ns, _last_monotonic_ns
    _last_wall_ns = None
    _last_monotonic_ns = None


def sample_clock(*, max_skew_ms: int) -> dict[str, Any]:
    """Read wall + monotonic clocks, compare to baseline, update baseline.

    Returns a dict with keys:
      status:               "first_probe_no_baseline" | "ok"
                          | "wall_clock_regression" | "monotonic_wall_skew_exceeded"
      wall_ns               current wall ts (ns)
      monotonic_ns          current monotonic ts (ns)
      wall_delta_ns         current_wall - last_wall (None on first probe)
      monotonic_delta_ns    current_mono - last_mono (None on first probe)
      skew_ms               |wall_delta - mono_delta| / 1e6 (None on first probe)
      max_skew_ms           the configured threshold echoed back

    Status precedence on subsequent probes:
      regression > skew_exceeded > ok
    Baseline updates after every probe regardless of status.
    """
    global _last_wall_ns, _last_monotonic_ns

    cur_wall = _now_wall_ns()
    cur_mono = _now_monotonic_ns()

    if _last_wall_ns is None or _last_monotonic_ns is None:
        result: dict[str, Any] = {
            "status": "first_probe_no_baseline",
            "wall_ns": cur_wall,
            "monotonic_ns": cur_mono,
            "wall_delta_ns": None,
            "monotonic_delta_ns": None,
            "skew_ms": None,
            "max_skew_ms": max_skew_ms,
        }
    else:
        wall_delta = cur_wall - _last_wall_ns
        mono_delta = cur_mono - _last_monotonic_ns
        skew_ms = abs(wall_delta - mono_delta) / 1_000_000.0
        if wall_delta < 0:
            status = "wall_clock_regression"
        elif skew_ms > max_skew_ms:
            status = "monotonic_wall_skew_exceeded"
        else:
            status = "ok"
        result = {
            "status": status,
            "wall_ns": cur_wall,
            "monotonic_ns": cur_mono,
            "wall_delta_ns": wall_delta,
            "monotonic_delta_ns": mono_delta,
            "skew_ms": skew_ms,
            "max_skew_ms": max_skew_ms,
        }

    _last_wall_ns = cur_wall
    _last_monotonic_ns = cur_mono
    return result


def check_ntp_synchronized(*, timeout_sec: float) -> dict[str, Any]:
    """Probe `timedatectl show -p NTPSynchronized --value`.

    Returns a dict with keys:
      status:    "ok" | "ntp_unsynced" | "ntp_binary_missing"
               | "ntp_probe_failed" | "ntp_probe_timeout"
      value:     parsed NTPSynchronized value (when probe completed)
      returncode/stderr/timeout_sec: forensic fields when applicable

    Does NOT raise — every error is mapped to a distinct status.
    """
    try:
        proc = _subprocess_run(
            ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError:
        return {"status": "ntp_binary_missing"}
    except subprocess.TimeoutExpired:
        return {"status": "ntp_probe_timeout", "timeout_sec": timeout_sec}

    if proc.returncode != 0:
        return {
            "status": "ntp_probe_failed",
            "returncode": int(proc.returncode),
            "stderr": (proc.stderr or "")[:200],
        }

    value = (proc.stdout or "").strip().lower()
    if value == "yes":
        return {"status": "ok", "value": value}
    return {"status": "ntp_unsynced", "value": value}
