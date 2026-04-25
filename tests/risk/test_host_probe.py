"""Phase 4.16 — host_probe unit tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from executor.risk import host_probe


def test_disk_pct_calculation(monkeypatch):
    class FakeUsage:
        total = 1000
        used = 850
        free = 150
    monkeypatch.setattr(host_probe.shutil, "disk_usage", lambda p: FakeUsage)
    assert host_probe.disk_pct("/whatever") == pytest.approx(85.0)


def test_disk_pct_zero_total(monkeypatch):
    class FakeUsage:
        total = 0
        used = 0
        free = 0
    monkeypatch.setattr(host_probe.shutil, "disk_usage", lambda p: FakeUsage)
    assert host_probe.disk_pct("/x") == 0.0


def test_inode_pct_calculation(monkeypatch):
    class FakeStat:
        f_files = 1000
        f_favail = 100  # 900 used
    monkeypatch.setattr(host_probe.os, "statvfs", lambda p: FakeStat)
    assert host_probe.inode_pct("/x") == pytest.approx(90.0)


def test_inode_pct_zero_total(monkeypatch):
    class FakeStat:
        f_files = 0
        f_favail = 0
    monkeypatch.setattr(host_probe.os, "statvfs", lambda p: FakeStat)
    assert host_probe.inode_pct("/x") == 0.0


def test_swap_pct_parses_meminfo(tmp_path: Path):
    p = tmp_path / "meminfo"
    p.write_text(
        "MemTotal:        1000000 kB\n"
        "MemFree:          500000 kB\n"
        "SwapTotal:       2000000 kB\n"
        "SwapFree:         500000 kB\n"
    )
    # Used 1_500_000 / 2_000_000 = 75%
    assert host_probe.swap_pct(str(p)) == pytest.approx(75.0)


def test_swap_pct_zero_total(tmp_path: Path):
    p = tmp_path / "meminfo"
    p.write_text("SwapTotal:             0 kB\nSwapFree:              0 kB\n")
    assert host_probe.swap_pct(str(p)) == 0.0


def test_swap_pct_missing_fields_raises(tmp_path: Path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:        1000 kB\n")
    with pytest.raises(RuntimeError, match="swap fields not found"):
        host_probe.swap_pct(str(p))


def test_sample_host_skips_optional_when_disabled(monkeypatch, tmp_path: Path):
    """rss/loadavg are not collected unless explicitly requested.

    Verifies the gate's optional-threshold contract (max=0 disables the
    probe entirely) translates into not-running these helpers.
    """
    class FakeUsage:
        total = 100
        used = 50
        free = 50
    class FakeStat:
        f_files = 100
        f_favail = 90
    p = tmp_path / "meminfo"
    p.write_text("SwapTotal:         100 kB\nSwapFree:          80 kB\n")
    monkeypatch.setattr(host_probe.shutil, "disk_usage", lambda x: FakeUsage)
    monkeypatch.setattr(host_probe.os, "statvfs", lambda x: FakeStat)

    def boom(*a, **kw):
        raise AssertionError("optional probe should not have been called")

    monkeypatch.setattr(host_probe, "rss_mb", boom)
    monkeypatch.setattr(host_probe, "loadavg_1m", boom)

    sample = host_probe.sample_host(
        check_rss=False, check_loadavg=False, meminfo_path=str(p)
    )
    assert "disk_pct" in sample
    assert "inode_pct" in sample
    assert "swap_pct" in sample
    assert "rss_mb" not in sample
    assert "loadavg_1m" not in sample


def test_sample_host_includes_optional_when_enabled(monkeypatch, tmp_path: Path):
    class FakeUsage:
        total = 100
        used = 25
        free = 75
    class FakeStat:
        f_files = 100
        f_favail = 50
    p = tmp_path / "meminfo"
    p.write_text("SwapTotal:         100 kB\nSwapFree:         100 kB\n")
    monkeypatch.setattr(host_probe.shutil, "disk_usage", lambda x: FakeUsage)
    monkeypatch.setattr(host_probe.os, "statvfs", lambda x: FakeStat)
    monkeypatch.setattr(host_probe, "rss_mb", lambda: 123.45)
    monkeypatch.setattr(host_probe, "loadavg_1m", lambda: 1.5)
    sample = host_probe.sample_host(
        check_rss=True, check_loadavg=True, meminfo_path=str(p)
    )
    assert sample["rss_mb"] == pytest.approx(123.45)
    assert sample["loadavg_1m"] == pytest.approx(1.5)


def test_sample_host_surfaces_probe_error(monkeypatch):
    """Underlying errors propagate so the gate can fail-closed/bypass."""
    def explode(*a, **kw):
        raise OSError("disk gone")
    monkeypatch.setattr(host_probe.shutil, "disk_usage", explode)
    with pytest.raises(OSError, match="disk gone"):
        host_probe.sample_host()
