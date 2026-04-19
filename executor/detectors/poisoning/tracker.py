"""
PoisoningTracker — binds a PoisoningDetector to runtime state.

Responsibilities:
- observe(market_id, prob): compute delta, call detector.check; on Anomaly,
  start a pause window for the market and publish ANOMALY_DETECTED.
- is_paused(market_id, now_ns): gate 2.6 polls this. Auto-resumes when
  pause_until < now_ns.

Pause semantics match the spec ("flagged markets auto-pause for 5min, then
re-check on next tick"): the pause is a cooldown, not a manual-resume state.
If the cooldown elapses and the next observation is still anomalous, another
pause window starts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Awaitable, Callable, Optional

from ...core.events import Event, EventType, Source
from ...core.logging import get_logger
from .base import Anomaly, PoisoningDetector


log = get_logger("executor.detectors.poisoning")


Publish = Callable[[Event], Awaitable[None]]


@dataclass
class _PauseRecord:
    since_ns: int
    until_ns: int
    anomaly: Anomaly


class PoisoningTracker:
    def __init__(
        self,
        detector: PoisoningDetector,
        *,
        pause_sec: int = 300,
        publish: Publish | None = None,
    ) -> None:
        self._detector = detector
        self._pause_sec = pause_sec
        self._publish = publish
        self._last_prob: dict[str, Decimal] = {}
        self._paused: dict[str, _PauseRecord] = {}

    @property
    def detector(self) -> PoisoningDetector:
        return self._detector

    def set_publish(self, publish: Publish) -> None:
        self._publish = publish

    def update_pause_sec(self, pause_sec: int) -> None:
        self._pause_sec = pause_sec

    async def observe(self, market_id: str, prob: Decimal) -> Optional[Anomaly]:
        """Feed a new price sample. Returns Anomaly if this sample tripped."""
        prev = self._last_prob.get(market_id)
        self._last_prob[market_id] = prob
        if prev is None:
            return None
        delta = prob - prev
        anomaly = await self._detector.check(market_id, delta)
        if anomaly is None:
            return None
        now_ns = anomaly.ts_ns
        self._paused[market_id] = _PauseRecord(
            since_ns=now_ns,
            until_ns=now_ns + self._pause_sec * 1_000_000_000,
            anomaly=anomaly,
        )
        log.warning(
            "risk.poisoning.anomaly",
            market_id=market_id,
            detector=anomaly.detector,
            score=anomaly.score,
            detail=anomaly.detail,
        )
        if self._publish is not None:
            await self._publish(
                Event.make(
                    EventType.ANOMALY_DETECTED,
                    source=Source.RISK,
                    market_id=market_id,
                    payload={
                        "detector": anomaly.detector,
                        "score": anomaly.score,
                        "detail": anomaly.detail,
                        "pause_until_ns": self._paused[market_id].until_ns,
                        "extra": anomaly.extra,
                    },
                )
            )
        return anomaly

    def is_paused(self, market_id: str, *, now_ns: int | None = None) -> tuple[bool, int, str]:
        """Returns (paused, until_ns, reason)."""
        now_ns = now_ns or time.time_ns()
        rec = self._paused.get(market_id)
        if rec is None:
            return False, 0, ""
        if rec.until_ns <= now_ns:
            # Cooldown elapsed. Keep the record (for history) but report un-paused.
            # Gate re-check happens on the next observe() call from the orderbook tick.
            return False, rec.until_ns, ""
        return True, rec.until_ns, rec.anomaly.detail

    def clear(self, market_id: str | None = None) -> None:
        if market_id is None:
            self._paused.clear()
        else:
            self._paused.pop(market_id, None)

    def snapshot(self) -> dict[str, Any]:
        return {
            "detector": self._detector.name,
            "paused_markets": {
                m: {"until_ns": r.until_ns, "score": r.anomaly.score}
                for m, r in self._paused.items()
            },
            "detector_state": self._detector.snapshot(),
        }
