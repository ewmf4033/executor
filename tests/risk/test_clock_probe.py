"""Phase 4.16 — clock_probe unit tests."""
from __future__ import annotations

import subprocess

import pytest

from executor.risk import clock_probe


@pytest.fixture(autouse=True)
def _reset():
    clock_probe.reset_baseline()
    yield
    clock_probe.reset_baseline()


def test_first_probe_returns_first_probe_no_baseline(monkeypatch):
    monkeypatch.setattr(clock_probe, "_now_wall_ns", lambda: 1_000_000_000)
    monkeypatch.setattr(clock_probe, "_now_monotonic_ns", lambda: 500_000_000)
    r = clock_probe.sample_clock(max_skew_ms=2000)
    assert r["status"] == "first_probe_no_baseline"
    assert r["wall_delta_ns"] is None
    assert r["monotonic_delta_ns"] is None
    assert r["skew_ms"] is None
    assert r["max_skew_ms"] == 2000
    assert r["wall_ns"] == 1_000_000_000
    assert r["monotonic_ns"] == 500_000_000


def test_detects_wall_clock_regression(monkeypatch):
    # First probe seeds baseline at wall=2e9, mono=1e9.
    walls = iter([2_000_000_000, 1_500_000_000])
    monos = iter([1_000_000_000, 1_500_000_000])  # +500ms mono
    monkeypatch.setattr(clock_probe, "_now_wall_ns", lambda: next(walls))
    monkeypatch.setattr(clock_probe, "_now_monotonic_ns", lambda: next(monos))

    first = clock_probe.sample_clock(max_skew_ms=2000)
    assert first["status"] == "first_probe_no_baseline"

    second = clock_probe.sample_clock(max_skew_ms=2000)
    # wall_delta = -500_000_000 (regression).
    assert second["status"] == "wall_clock_regression"
    assert second["wall_delta_ns"] < 0
    assert second["monotonic_delta_ns"] == 500_000_000


def test_detects_monotonic_wall_skew_above_threshold(monkeypatch):
    # Baseline then a probe where wall jumped 5s but mono only 1s →
    # skew = 4000ms > 2000ms threshold.
    walls = iter([1_000_000_000, 6_000_000_000])
    monos = iter([1_000_000_000, 2_000_000_000])
    monkeypatch.setattr(clock_probe, "_now_wall_ns", lambda: next(walls))
    monkeypatch.setattr(clock_probe, "_now_monotonic_ns", lambda: next(monos))

    clock_probe.sample_clock(max_skew_ms=2000)
    r = clock_probe.sample_clock(max_skew_ms=2000)
    assert r["status"] == "monotonic_wall_skew_exceeded"
    assert r["skew_ms"] == pytest.approx(4000.0)
    assert r["max_skew_ms"] == 2000


def test_reset_baseline_clears_state(monkeypatch):
    monkeypatch.setattr(clock_probe, "_now_wall_ns", lambda: 1_000_000_000)
    monkeypatch.setattr(clock_probe, "_now_monotonic_ns", lambda: 1_000_000_000)
    clock_probe.sample_clock(max_skew_ms=2000)
    # Baseline now set; next probe would be "ok".
    assert clock_probe._last_wall_ns is not None
    assert clock_probe._last_monotonic_ns is not None

    clock_probe.reset_baseline()
    assert clock_probe._last_wall_ns is None
    assert clock_probe._last_monotonic_ns is None

    # Confirm next sample is treated as the first.
    r = clock_probe.sample_clock(max_skew_ms=2000)
    assert r["status"] == "first_probe_no_baseline"


def _fake_completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["timedatectl"], returncode=returncode, stdout=stdout, stderr=stderr,
    )


def test_timedatectl_synchronized_yes_is_ok(monkeypatch):
    def fake_run(*a, **kw):
        return _fake_completed(stdout="yes\n", returncode=0)
    monkeypatch.setattr(clock_probe, "_subprocess_run", fake_run)
    r = clock_probe.check_ntp_synchronized(timeout_sec=2.0)
    assert r["status"] == "ok"
    assert r["value"] == "yes"


def test_timedatectl_unsynchronized_is_unsynced(monkeypatch):
    def fake_run(*a, **kw):
        return _fake_completed(stdout="no\n", returncode=0)
    monkeypatch.setattr(clock_probe, "_subprocess_run", fake_run)
    r = clock_probe.check_ntp_synchronized(timeout_sec=2.0)
    assert r["status"] == "ntp_unsynced"
    assert r["value"] == "no"


def test_timedatectl_missing_nonzero_timeout_distinct_statuses(monkeypatch):
    # Missing binary.
    def raise_fnf(*a, **kw):
        raise FileNotFoundError("no timedatectl")
    monkeypatch.setattr(clock_probe, "_subprocess_run", raise_fnf)
    r1 = clock_probe.check_ntp_synchronized(timeout_sec=2.0)
    assert r1["status"] == "ntp_binary_missing"

    # Nonzero return code.
    def fake_run_failed(*a, **kw):
        return _fake_completed(returncode=1, stderr="permission denied")
    monkeypatch.setattr(clock_probe, "_subprocess_run", fake_run_failed)
    r2 = clock_probe.check_ntp_synchronized(timeout_sec=2.0)
    assert r2["status"] == "ntp_probe_failed"
    assert r2["returncode"] == 1
    assert "permission denied" in r2["stderr"]

    # Timeout.
    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="timedatectl", timeout=2.0)
    monkeypatch.setattr(clock_probe, "_subprocess_run", raise_timeout)
    r3 = clock_probe.check_ntp_synchronized(timeout_sec=2.0)
    assert r3["status"] == "ntp_probe_timeout"
    assert r3["timeout_sec"] == 2.0
