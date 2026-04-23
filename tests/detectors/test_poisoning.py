"""0g — ZScoreDetector + PoisoningTracker."""
from __future__ import annotations

import time
from decimal import Decimal

import pytest

from executor.core.events import EventType
from executor.core.event_bus import EventBus
from executor.detectors.poisoning import (
    PoisoningTracker,
    ZScoreDetector,
    build_detector,
    list_detectors,
)




async def test_zscore_flags_outlier():
    det = ZScoreDetector(window_sec=3600, z_threshold=2.0, min_samples=20)
    # Prime the window with quiet deltas.
    for i in range(25):
        a = await det.check("MKT", Decimal("0.0001"))
        assert a is None
    # A wildly different delta.
    a = await det.check("MKT", Decimal("0.5"))
    assert a is not None
    assert a.detector == "zscore"
    assert abs(a.score) > 2.0
    assert "|z|" in a.detail


async def test_zscore_below_threshold_passes():
    det = ZScoreDetector(window_sec=3600, z_threshold=10.0, min_samples=10)
    for _ in range(15):
        assert await det.check("MKT", Decimal("0.0001")) is None


async def test_zscore_min_samples_gate():
    det = ZScoreDetector(window_sec=3600, z_threshold=1.0, min_samples=50)
    # Huge first delta — but min_samples not met → no flag.
    for i in range(10):
        assert await det.check("MKT", Decimal("0.5")) is None


async def test_registry_builds_default_zscore():
    det = build_detector("zscore", window_sec=100, z_threshold=4.0, min_samples=5)
    assert det.name == "zscore"
    assert "zscore" in list_detectors()


async def test_registry_unknown_raises():
    with pytest.raises(ValueError):
        build_detector("not-a-detector")


async def test_tracker_auto_pauses_then_resumes_after_cooldown():
    det = ZScoreDetector(window_sec=3600, z_threshold=2.0, min_samples=20)
    tracker = PoisoningTracker(det, pause_sec=1)  # 1s cooldown for test speed
    # Prime.
    await tracker.observe("MKT", Decimal("0.500"))
    for i in range(25):
        await tracker.observe("MKT", Decimal("0.500") + Decimal("0.0001") * i)
    anomaly = await tracker.observe("MKT", Decimal("0.95"))
    assert anomaly is not None

    paused, until_ns, _ = tracker.is_paused("MKT")
    assert paused is True
    # Before cooldown.
    import asyncio
    await asyncio.sleep(1.1)
    paused, _, _ = tracker.is_paused("MKT")
    assert paused is False


async def test_tracker_emits_anomaly_detected_event():
    det = ZScoreDetector(window_sec=3600, z_threshold=2.0, min_samples=20)
    bus = EventBus()
    await bus.start()
    got = []

    async def sink(e):
        got.append(e)

    await bus.subscribe("sink", on_event=sink)
    tracker = PoisoningTracker(det, pause_sec=60, publish=bus.publish)
    await tracker.observe("MKT", Decimal("0.5"))
    for i in range(25):
        await tracker.observe("MKT", Decimal("0.5") + Decimal("0.0001") * i)
    await tracker.observe("MKT", Decimal("0.95"))
    import asyncio
    await asyncio.sleep(0.05)
    await bus.stop()
    anomaly_events = [e for e in got if e.event_type == EventType.ANOMALY_DETECTED]
    assert len(anomaly_events) == 1
    payload = anomaly_events[0].payload
    assert payload["detector"] == "zscore"
    assert payload["pause_until_ns"] > time.time_ns()


async def test_pluggable_custom_detector_via_registry():
    """A third-party detector can register without touching gate code."""
    from executor.detectors.poisoning.base import Anomaly, PoisoningDetector
    from executor.detectors.poisoning.registry import register_detector, build_detector

    class AlwaysFlag(PoisoningDetector):
        name = "alwaysflag"
        async def check(self, market_id, prob_delta):
            if abs(prob_delta) < Decimal("0.001"):
                return None
            return Anomaly(market_id=market_id, detector=self.name,
                           score=1.0, ts_ns=time.time_ns(), detail="custom")

    # Register under a unique name (avoid collision between test runs).
    unique = f"test_alwaysflag_{time.time_ns()}"
    register_detector(unique, lambda **kw: AlwaysFlag())
    det = build_detector(unique)
    a = await det.check("M", Decimal("0.5"))
    assert a is not None and a.detector == "alwaysflag"
