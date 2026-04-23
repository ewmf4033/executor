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


# ---------------------------------------------------------------------------
# Phase 4.12 — 0g hardening: input validation, exception containment, bounds.
# ---------------------------------------------------------------------------


async def _sink_bus():
    """Return (bus, events_list) with a subscribed sink that appends."""
    bus = EventBus()
    await bus.start()
    got: list = []

    async def sink(e):
        got.append(e)

    await bus.subscribe("sink", on_event=sink)
    return bus, got


# Finding #3 — input validation


async def test_tracker_rejects_empty_market_id():
    det = ZScoreDetector(window_sec=3600, z_threshold=5.0, min_samples=20)
    bus, got = await _sink_bus()
    tracker = PoisoningTracker(det, pause_sec=60, publish=bus.publish)
    result = await tracker.observe("", Decimal("0.5"))
    import asyncio
    await asyncio.sleep(0.05)
    await bus.stop()
    assert result is None
    rejected = [e for e in got if e.event_type == EventType.POISONING_INPUT_REJECTED]
    assert len(rejected) == 1
    assert rejected[0].payload["reason"] == "empty_market_id"
    # No state mutation.
    assert tracker.snapshot()["markets_tracked"] == 0


async def test_tracker_rejects_overlong_market_id():
    det = ZScoreDetector(window_sec=3600, z_threshold=5.0, min_samples=20)
    bus, got = await _sink_bus()
    tracker = PoisoningTracker(det, pause_sec=60, publish=bus.publish)
    long_id = "M" * 129
    result = await tracker.observe(long_id, Decimal("0.5"))
    import asyncio
    await asyncio.sleep(0.05)
    await bus.stop()
    assert result is None
    rejected = [e for e in got if e.event_type == EventType.POISONING_INPUT_REJECTED]
    assert len(rejected) == 1
    assert rejected[0].payload["reason"] == "overlong_market_id"
    assert tracker.snapshot()["markets_tracked"] == 0


async def test_tracker_rejects_non_finite_prob():
    det = ZScoreDetector(window_sec=3600, z_threshold=5.0, min_samples=20)
    bus, got = await _sink_bus()
    tracker = PoisoningTracker(det, pause_sec=60, publish=bus.publish)
    for bad in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
        assert await tracker.observe("MKT", bad) is None
    import asyncio
    await asyncio.sleep(0.05)
    await bus.stop()
    rejected = [e for e in got if e.event_type == EventType.POISONING_INPUT_REJECTED]
    assert len(rejected) == 3
    assert all(e.payload["reason"] == "non_finite_prob" for e in rejected)
    assert tracker.snapshot()["markets_tracked"] == 0


async def test_tracker_rejects_prob_out_of_range():
    det = ZScoreDetector(window_sec=3600, z_threshold=5.0, min_samples=20)
    bus, got = await _sink_bus()
    tracker = PoisoningTracker(det, pause_sec=60, publish=bus.publish)
    assert await tracker.observe("MKT", Decimal("-0.1")) is None
    assert await tracker.observe("MKT", Decimal("1.1")) is None
    import asyncio
    await asyncio.sleep(0.05)
    await bus.stop()
    rejected = [e for e in got if e.event_type == EventType.POISONING_INPUT_REJECTED]
    assert len(rejected) == 2
    assert all(e.payload["reason"] == "prob_out_of_range" for e in rejected)


async def test_tracker_rejects_negative_ts_ns():
    det = ZScoreDetector(window_sec=3600, z_threshold=5.0, min_samples=20)
    bus, got = await _sink_bus()
    tracker = PoisoningTracker(det, pause_sec=60, publish=bus.publish)
    # Seed so that an observe call with negative ts_ns reaches the validator
    # (validators run before delta compute, so no prior state needed).
    assert await tracker.observe("MKT", Decimal("0.5"), ts_ns=-1) is None
    assert await tracker.observe("MKT", Decimal("0.5"), ts_ns=0) is None
    import asyncio
    await asyncio.sleep(0.05)
    await bus.stop()
    rejected = [e for e in got if e.event_type == EventType.POISONING_INPUT_REJECTED]
    assert len(rejected) == 2
    assert all(e.payload["reason"] == "non_positive_ts_ns" for e in rejected)


# Finding #4 — exception containment


async def test_tracker_containment_on_detector_raise():
    from executor.detectors.poisoning.base import PoisoningDetector as PD

    class ExplodingDetector(PD):
        name = "exploder"
        calls = 0
        async def check(self, market_id, prob_delta):
            ExplodingDetector.calls += 1
            raise RuntimeError("boom")

    det = ExplodingDetector()
    bus, got = await _sink_bus()
    tracker = PoisoningTracker(det, pause_sec=60, publish=bus.publish)
    # First observe seeds prev; no detector call yet.
    await tracker.observe("MKT", Decimal("0.5"))
    # Second observe triggers detector.check(), which raises.
    result = await tracker.observe("MKT", Decimal("0.6"))
    import asyncio
    await asyncio.sleep(0.05)
    await bus.stop()
    # (a) observe() did not raise; returned None.
    assert result is None
    # (b) POISONING_DETECTOR_ERROR event emitted.
    errors = [e for e in got if e.event_type == EventType.POISONING_DETECTOR_ERROR]
    assert len(errors) == 1
    assert errors[0].payload["exception_type"] == "RuntimeError"
    assert errors[0].payload["detector"] == "exploder"
    # (c) market is paused (fail-closed).
    paused, until_ns, reason = tracker.is_paused("MKT")
    assert paused is True
    assert until_ns > 0
    assert "detector_error" in reason


async def test_tracker_containment_on_publish_failure():
    det = ZScoreDetector(window_sec=3600, z_threshold=5.0, min_samples=20)

    async def bad_publish(event):
        raise RuntimeError("bus is down")

    tracker = PoisoningTracker(det, pause_sec=60, publish=bad_publish)
    # An invalid input triggers publish; must not raise.
    result = await tracker.observe("", Decimal("0.5"))
    assert result is None
    # Valid input path also invokes publish (via future anomaly) — exercise
    # the happy path to confirm no regression, but the core assertion above
    # is that observe() swallowed the publish RuntimeError.


# Finding #1 — bounded state (LRU)


async def test_zscore_evicts_oldest_when_over_cap():
    from executor.detectors.poisoning.zscore import MAX_MARKETS
    det = ZScoreDetector(window_sec=3600, z_threshold=5.0, min_samples=20)
    # Fill exactly MAX_MARKETS.
    for i in range(MAX_MARKETS):
        await det.check(f"M{i}", Decimal("0.0001"))
    snap = det.snapshot()
    assert snap["markets_tracked"] == MAX_MARKETS
    assert snap["evictions_total"] == 0
    # Insert one more — oldest ("M0") should be evicted.
    await det.check("M_NEW", Decimal("0.0001"))
    snap = det.snapshot()
    assert snap["markets_tracked"] == MAX_MARKETS
    assert snap["evictions_total"] == 1
    assert "M_NEW" in snap["per_market"]
    assert "M0" not in snap["per_market"]


async def test_tracker_prunes_expired_pauses():
    from executor.detectors.poisoning.base import Anomaly, PoisoningDetector as PD

    # Detector that flags deterministically on first real delta.
    class FlagOnce(PD):
        name = "flagonce"
        async def check(self, market_id, prob_delta):
            return Anomaly(
                market_id=market_id, detector=self.name, score=10.0,
                ts_ns=time.time_ns(), detail="flag",
            )

    det = FlagOnce()
    tracker = PoisoningTracker(det, pause_sec=1)
    # Seed then flag market A.
    await tracker.observe("A", Decimal("0.5"))
    anomaly = await tracker.observe("A", Decimal("0.6"))
    assert anomaly is not None
    assert "A" in tracker._paused
    # Wait past cooldown so A's pause_until_ns < now_ns.
    import asyncio
    await asyncio.sleep(1.1)
    # Clear A's last_prob so prune-condition (no recent activity) fires.
    tracker._last_prob.pop("A", None)
    # Unrelated observe on market B triggers opportunistic prune.
    await tracker.observe("B", Decimal("0.5"))
    # A's expired pause record should be pruned.
    assert "A" not in tracker._paused
    snap = tracker.snapshot()
    assert snap["prune_cycles"] >= 1
