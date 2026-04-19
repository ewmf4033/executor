"""
Venue-health tracker used by Gate 2.5.

Counts VenueDown/RateLimited incidents per venue within a sliding window.
If >= trip_threshold within window_sec, pauses that venue for pause_sec.
Subsequent tickets hit Gate 2.5 and reject with VENUE_HEALTH_TRIPPED.
Auto-resume on next check once pause_until < now.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

from ..core.logging import get_logger


log = get_logger("executor.risk.venue_health")


@dataclass
class _VenueRecord:
    incidents: Deque[int]           # ts_ns of recent incidents
    paused_until_ns: int = 0


class VenueHealth:
    def __init__(self, *, window_sec: int, trip_threshold: int, pause_sec: int) -> None:
        self.window_sec = window_sec
        self.trip_threshold = trip_threshold
        self.pause_sec = pause_sec
        self._venues: dict[str, _VenueRecord] = {}

    def update_from_config(self, *, window_sec: int, trip_threshold: int, pause_sec: int) -> None:
        self.window_sec = window_sec
        self.trip_threshold = trip_threshold
        self.pause_sec = pause_sec

    def record_incident(self, venue: str, *, now_ns: int | None = None) -> bool:
        """Record a VenueDown/RateLimited. Returns True if this incident
        tripped the pause (first time)."""
        now_ns = now_ns or time.time_ns()
        rec = self._venues.setdefault(venue, _VenueRecord(incidents=deque()))
        rec.incidents.append(now_ns)
        self._trim(rec, now_ns)
        if len(rec.incidents) >= self.trip_threshold and rec.paused_until_ns <= now_ns:
            rec.paused_until_ns = now_ns + self.pause_sec * 1_000_000_000
            log.warning(
                "risk.venue_health.tripped",
                venue=venue,
                incidents=len(rec.incidents),
                pause_until_ns=rec.paused_until_ns,
            )
            return True
        return False

    def is_paused(self, venue: str, *, now_ns: int | None = None) -> tuple[bool, int]:
        """Returns (paused, paused_until_ns). paused_until_ns=0 if never paused."""
        now_ns = now_ns or time.time_ns()
        rec = self._venues.get(venue)
        if rec is None:
            return False, 0
        self._trim(rec, now_ns)
        if rec.paused_until_ns > now_ns:
            return True, rec.paused_until_ns
        return False, rec.paused_until_ns

    def _trim(self, rec: _VenueRecord, now_ns: int) -> None:
        cutoff = now_ns - self.window_sec * 1_000_000_000
        while rec.incidents and rec.incidents[0] < cutoff:
            rec.incidents.popleft()

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {
            v: {"incidents": len(r.incidents), "paused_until_ns": r.paused_until_ns}
            for v, r in self._venues.items()
        }
