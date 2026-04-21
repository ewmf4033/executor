"""
WindowAdverseSelectionDetector — real Phase 4 0e implementation.

For each fill on a venue, observe the venue's mid-price at t+30s, t+60s,
t+300s after the fill. Compute "adverse movement" relative to the fill
direction (BUY filled long means we lose if mid drops; SELL filled short
means we lose if mid rises).

Tracks per-venue rolling window of the most recent N fills. If `>=
adverse_threshold` of those fills moved adversely by `>= move_threshold_sigma`
standard deviations of recent venue mid changes, emit ADVERSE_SELECTION_DETECTED
and pause the venue for `pause_sec` seconds (soft pause — stop new intents
to that venue).

Implementation notes:
- Sigma is estimated from the rolling-window mid changes themselves; if the
  window is too small to estimate sigma, we skip the check.
- The detector exposes is_flagged(strategy_id, market_id) AND is_venue_paused(venue)
  so existing Gate 3 keeps working AND a new soft-pause check can run.
- "Adverse movement" is measured at t+60s (the middle horizon) for the
  pause-trigger decision; the other horizons (30s, 300s) are recorded for
  attribution-style analysis but do not trigger pauses on their own.
"""
from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Awaitable, Callable, Deque, Optional

from typing import TYPE_CHECKING

from ...core.events import Event, EventType, Source
from ...core.logging import get_logger
from ...core.types import Side
from .base import AdverseSelectionDetector, AdverseSelectionFlag

if TYPE_CHECKING:
    from ...risk.state import RiskState


# Sentinel used for durable (no-TTL) pauses. ~2262-04-11; larger than any
# realistic now_ns so is_venue_paused returns True indefinitely.
_DURABLE_PAUSED_SENTINEL_NS = 2**62


log = get_logger("executor.detectors.adverse_selection")


Publish = Callable[[Event], Awaitable[None]]


@dataclass
class _FillRecord:
    venue: str
    market_id: str
    side: Side
    fill_price: Decimal
    ts_ns: int
    mid_30: Decimal | None = None
    mid_60: Decimal | None = None
    mid_300: Decimal | None = None
    adverse_60_sigma: float | None = None  # cached score
    decided: bool = False                   # whether this fill counted in the rolling check


@dataclass
class _VenueWindow:
    fills: Deque[_FillRecord] = field(default_factory=deque)
    mids: Deque[Decimal] = field(default_factory=lambda: deque(maxlen=200))
    paused_until_ns: int = 0
    last_anomaly_score: float = 0.0


class WindowAdverseSelectionDetector(AdverseSelectionDetector):
    """
    Phase 4 implementation. Implements the gate-3 interface (is_flagged) and
    adds venue-pause semantics consumed by orchestration code.
    """

    def __init__(
        self,
        *,
        window: int = 20,
        adverse_threshold: float = 0.60,
        move_threshold_sigma: float = 2.0,
        pause_sec: int = 300,
        publish: Publish | None = None,
        state: "RiskState | None" = None,
    ) -> None:
        self.window = window
        self.adverse_threshold = adverse_threshold
        self.move_threshold_sigma = move_threshold_sigma
        self.pause_sec = pause_sec
        self._publish = publish
        self._venues: dict[str, _VenueWindow] = {}
        # Markets the gate should reject. Keyed by (strategy_id_or_*, market_id).
        # We only track per-market flags here; broader strategy/market scoping
        # is the caller's job.
        self._flagged_markets: set[str] = set()
        # Phase 4.9 Item 2: durable pause persistence. When a RiskState is
        # attached, every pause/clear mirrors into adverse_selection_pauses
        # so pauses survive daemon restart.
        self._state: "RiskState | None" = state

    def set_state(self, state: "RiskState") -> None:
        """Late-binding for DaemonService: detector is constructed before
        RiskState.load() completes."""
        self._state = state

    def load_from_state(self, state: "RiskState") -> None:
        """Rehydrate in-memory pause set from the durable table. Called by
        DaemonService at startup so venues paused before a restart stay paused.
        Rehydrated pauses have no TTL — paused_until_ns is a far-future
        sentinel so is_venue_paused stays True until clear_venue() runs."""
        self._state = state
        rows = state.list_adverse_pauses()
        for row in rows:
            venue = row["venue"]
            rec = self._venues.setdefault(venue, _VenueWindow())
            rec.paused_until_ns = _DURABLE_PAUSED_SENTINEL_NS
            log.warning(
                "risk.adverse_selection.rehydrated",
                venue=venue,
                paused_at_ns=row["paused_at_ns"],
                reason=row["reason"],
                source_market_id=row["source_market_id"],
            )

    # ------------------------------------------------------------------
    # Gate-3 interface
    # ------------------------------------------------------------------

    def is_flagged(self, *, strategy_id: str, market_id: str) -> bool:
        return market_id in self._flagged_markets

    # ------------------------------------------------------------------
    # New surface for venue-soft-pause (consumed by orchestration loop)
    # ------------------------------------------------------------------

    def set_publish(self, publish: Publish) -> None:
        self._publish = publish

    def venue_pause_detail(self, venue: str, *, now_ns: int | None = None) -> tuple[bool, int]:
        """Full pause detail: (is_paused, paused_until_ns). Used by tests
        and observers that want the expiry timestamp."""
        rec = self._venues.get(venue)
        if rec is None:
            return False, 0
        now_ns = now_ns or time.time_ns()
        if rec.paused_until_ns > now_ns:
            return True, rec.paused_until_ns
        return False, rec.paused_until_ns

    def is_venue_paused(self, venue: str) -> bool:
        """Gate-3 interface — bool-only per AdverseSelectionDetector."""
        paused, _ = self.venue_pause_detail(venue)
        return paused

    def clear_venue(self, venue: str) -> None:
        rec = self._venues.get(venue)
        if rec is not None:
            rec.paused_until_ns = 0
            rec.last_anomaly_score = 0.0
        # Phase 4.9 Item 2: remove durable pause row if present.
        if self._state is not None:
            try:
                self._state.clear_adverse_pause(venue)
            except Exception as exc:
                log.warning(
                    "risk.adverse_selection.clear_persist_failed",
                    venue=venue,
                    error=str(exc),
                )

    # ------------------------------------------------------------------
    # Observation feeds
    # ------------------------------------------------------------------

    def observe_mid(self, venue: str, mid: Decimal) -> None:
        rec = self._venues.setdefault(venue, _VenueWindow())
        rec.mids.append(Decimal(str(mid)))

    async def observe_fill(
        self,
        *,
        venue: str,
        market_id: str,
        side: Side,
        fill_price: Decimal,
        fill_ts_ns: int,
    ) -> None:
        rec = self._venues.setdefault(venue, _VenueWindow())
        f = _FillRecord(
            venue=venue,
            market_id=market_id,
            side=side,
            fill_price=Decimal(str(fill_price)),
            ts_ns=int(fill_ts_ns),
        )
        rec.fills.append(f)
        # Trim to rolling window of `self.window`.
        while len(rec.fills) > self.window:
            rec.fills.popleft()

    async def update_mark(
        self,
        *,
        venue: str,
        market_id: str,
        mid: Decimal,
        now_ns: int | None = None,
    ) -> Optional[AdverseSelectionFlag]:
        """
        Call this on every mid-tick. We attach the mid sample to any unfilled
        post-fill horizons (30s/60s/300s) and, if a record's 60s horizon just
        elapsed, run the rolling-window check for this venue.
        Returns an AdverseSelectionFlag if a NEW pause was just triggered.
        """
        now_ns = now_ns or time.time_ns()
        rec = self._venues.get(venue)
        if rec is None:
            return None
        rec.mids.append(Decimal(str(mid)))
        flag: Optional[AdverseSelectionFlag] = None
        for f in rec.fills:
            if f.market_id != market_id:
                continue
            elapsed_sec = (now_ns - f.ts_ns) / 1e9
            if f.mid_30 is None and elapsed_sec >= 30:
                f.mid_30 = mid
            if f.mid_60 is None and elapsed_sec >= 60:
                f.mid_60 = mid
                # The 60s checkpoint is what counts the fill in the rolling
                # decision; compute it now and run the venue-level threshold.
                f.adverse_60_sigma = self._adverse_sigma(rec, f, mid)
                f.decided = True
                triggered = self._check_threshold(rec, venue, market_id, now_ns)
                if triggered:
                    flag = triggered
            if f.mid_300 is None and elapsed_sec >= 300:
                f.mid_300 = mid
        return flag

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _adverse_sigma(self, rec: _VenueWindow, f: _FillRecord, mid_60: Decimal) -> float:
        """How many sigma adverse vs the venue's mid-change distribution?"""
        # Mid changes for sigma estimate.
        diffs: list[float] = []
        prev = None
        for m in rec.mids:
            if prev is not None:
                diffs.append(float(m - prev))
            prev = m
        if len(diffs) < 5:
            return 0.0
        sigma = statistics.pstdev(diffs)
        if sigma <= 0:
            return 0.0
        # BUY: adverse if mid_60 < fill_price (post-fill price drop hurts long).
        # SELL: adverse if mid_60 > fill_price.
        delta = float(mid_60 - f.fill_price)
        adverse_signed = -delta if f.side == Side.BUY else delta
        return adverse_signed / sigma

    def _check_threshold(
        self, rec: _VenueWindow, venue: str, market_id: str, now_ns: int
    ) -> Optional[AdverseSelectionFlag]:
        decided = [f for f in rec.fills if f.decided]
        if len(decided) < self.window:
            return None
        adverse_count = sum(
            1 for f in decided
            if (f.adverse_60_sigma or 0.0) >= self.move_threshold_sigma
        )
        ratio = adverse_count / len(decided)
        if ratio < self.adverse_threshold:
            return None
        # Already paused? Don't re-fire.
        if rec.paused_until_ns > now_ns:
            return None
        # Phase 4.9 Item 2: pauses are durable (no automatic TTL). A paused
        # venue stays paused until an operator explicitly calls clear_venue.
        # The sentinel keeps is_venue_paused() True across any realistic
        # wall-clock, including tests that pass an old now_ns.
        rec.paused_until_ns = _DURABLE_PAUSED_SENTINEL_NS
        rec.last_anomaly_score = ratio
        # Flag the market that triggered the latest decision for the gate too.
        self._flagged_markets.add(market_id)
        # Phase 4.9 Item 2: persist the pause so it survives daemon restart.
        if self._state is not None:
            try:
                self._state.save_adverse_pause(
                    venue=venue,
                    paused_at_ns=now_ns,
                    reason=(
                        f"{adverse_count}/{len(decided)} fills "
                        f">= {self.move_threshold_sigma}σ adverse"
                    ),
                    source_market_id=market_id,
                )
            except Exception as exc:
                log.warning(
                    "risk.adverse_selection.persist_failed",
                    venue=venue,
                    error=str(exc),
                )
        flag = AdverseSelectionFlag(
            strategy_id="*",
            market_id=market_id,
            reason=(
                f"adverse_selection: {adverse_count}/{len(decided)} fills "
                f">= {self.move_threshold_sigma}σ adverse; venue {venue} paused "
                f"{self.pause_sec}s"
            ),
            ts_ns=now_ns,
        )
        log.warning(
            "risk.adverse_selection.tripped",
            venue=venue,
            market_id=market_id,
            ratio=ratio,
            window=len(decided),
        )
        return flag

    async def emit_flag_event(self, flag: AdverseSelectionFlag, *, venue: str) -> None:
        if self._publish is None:
            return
        await self._publish(
            Event.make(
                EventType.ANOMALY_DETECTED,
                source=Source.RISK,
                venue=venue,
                market_id=flag.market_id,
                payload={
                    "kind": "ADVERSE_SELECTION_DETECTED",
                    "reason": flag.reason,
                    "pause_sec": self.pause_sec,
                },
            )
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for v, rec in self._venues.items():
            decided = [f for f in rec.fills if f.decided]
            adverse = sum(
                1 for f in decided
                if (f.adverse_60_sigma or 0.0) >= self.move_threshold_sigma
            )
            out[v] = {
                "fills_in_window": len(rec.fills),
                "decided_fills": len(decided),
                "adverse_count": adverse,
                "paused_until_ns": rec.paused_until_ns,
                "last_score": rec.last_anomaly_score,
            }
        return out
