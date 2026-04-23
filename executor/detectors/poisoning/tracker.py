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

Phase 4.12 hardening (Codex Review 7 v2):
- Input validation at observe() boundary: market_id non-empty + bounded,
  prob Decimal + finite + in [0,1], ts_ns positive. Invalid inputs emit
  POISONING_INPUT_REJECTED and return early.
- Exception containment: detector.check() and publish() wrapped; detector
  exceptions fail-close the market (auto-pause) and emit
  POISONING_DETECTOR_ERROR. observe() never raises to caller.
- Bounded state: _last_prob and _paused LRU-capped at MAX_MARKETS; expired
  pauses pruned opportunistically on each observe().
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Awaitable, Callable, Optional

from ...core.events import Event, EventType, Source
from ...core.logging import get_logger
from .base import Anomaly, PoisoningDetector


log = get_logger("executor.detectors.poisoning")


Publish = Callable[[Event], Awaitable[None]]


# Hard caps: defense against adversarial/malformed feeds allocating
# unbounded market_ids. At ~10k active Kalshi markets peak, 10_000 is
# ~10x headroom.
MAX_MARKETS = 10_000
MAX_MARKET_ID_LEN = 128


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
        # OrderedDicts give us O(1) LRU semantics via move_to_end + popitem(last=False).
        self._last_prob: "OrderedDict[str, Decimal]" = OrderedDict()
        self._paused: "OrderedDict[str, _PauseRecord]" = OrderedDict()
        self._evictions_total = 0
        self._prune_cycles = 0
        self._inputs_rejected = 0
        self._detector_errors = 0

    @property
    def detector(self) -> PoisoningDetector:
        return self._detector

    def set_publish(self, publish: Publish) -> None:
        self._publish = publish

    def update_pause_sec(self, pause_sec: int) -> None:
        self._pause_sec = pause_sec

    async def _emit(self, event: Event) -> None:
        """Publish an event, swallowing any exception from the bus.

        Audit-log failure must never crash the ingestion path.
        """
        if self._publish is None:
            return
        try:
            await self._publish(event)
        except Exception:  # pragma: no cover - defensive
            log.exception("risk.poisoning.publish_failed", event_type=str(event.event_type))

    async def _reject(self, market_id: str, reason: str, detail: dict[str, Any]) -> None:
        self._inputs_rejected += 1
        log.warning("risk.poisoning.input_rejected", market_id=market_id, reason=reason, **detail)
        await self._emit(
            Event.make(
                EventType.POISONING_INPUT_REJECTED,
                source=Source.RISK,
                market_id=market_id if market_id else None,
                payload={"reason": reason, **detail},
            )
        )

    def _prune_expired_pauses(self, now_ns: int) -> None:
        """Opportunistic prune of pause records whose cooldown has elapsed
        and which have no recent price activity tracked.

        Keeps _paused bounded in the presence of many short-lived markets.
        """
        if not self._paused:
            return
        self._prune_cycles += 1
        expired = [
            m for m, r in self._paused.items()
            if r.until_ns < now_ns and m not in self._last_prob
        ]
        for m in expired:
            self._paused.pop(m, None)

    def _touch_last_prob(self, market_id: str, prob: Decimal) -> None:
        """Record prob with LRU semantics; evict oldest if over cap."""
        if market_id in self._last_prob:
            self._last_prob.move_to_end(market_id)
            self._last_prob[market_id] = prob
            return
        if len(self._last_prob) >= MAX_MARKETS:
            self._last_prob.popitem(last=False)
            self._evictions_total += 1
        self._last_prob[market_id] = prob

    def _touch_paused(self, market_id: str, rec: _PauseRecord) -> None:
        """Insert pause record with LRU semantics."""
        if market_id in self._paused:
            self._paused.move_to_end(market_id)
            self._paused[market_id] = rec
            return
        if len(self._paused) >= MAX_MARKETS:
            self._paused.popitem(last=False)
            self._evictions_total += 1
        self._paused[market_id] = rec

    async def observe(
        self,
        market_id: str,
        prob: Decimal,
        ts_ns: int | None = None,
    ) -> Optional[Anomaly]:
        """Feed a new price sample. Returns Anomaly if this sample tripped.

        Never raises. Invalid inputs are rejected (audit event + early return).
        Detector exceptions fail-close the market (auto-pause) but do not
        propagate.
        """
        now_ns = ts_ns if ts_ns is not None else time.time_ns()

        # --- Input validation (Finding #3) --------------------------------
        # market_id: non-empty str, bounded length.
        if not isinstance(market_id, str) or not market_id:
            await self._reject(market_id if isinstance(market_id, str) else "", "empty_market_id", {})
            return None
        if len(market_id) > MAX_MARKET_ID_LEN:
            await self._reject(market_id[:MAX_MARKET_ID_LEN], "overlong_market_id", {"len": len(market_id)})
            return None
        # prob: Decimal type enforced strictly — caller bug if not.
        if not isinstance(prob, Decimal):
            raise TypeError(f"PoisoningTracker.observe: prob must be Decimal, got {type(prob).__name__}")
        # prob: must be finite (not NaN, not ±Infinity).
        if not prob.is_finite():
            await self._reject(market_id, "non_finite_prob", {"prob_repr": str(prob)})
            return None
        # prob: must be in [0, 1] inclusive (Kalshi contract probability).
        if prob < Decimal(0) or prob > Decimal(1):
            await self._reject(market_id, "prob_out_of_range", {"prob": str(prob)})
            return None
        # ts_ns: positive int.
        if not isinstance(ts_ns, int) and ts_ns is not None:
            await self._reject(market_id, "ts_ns_not_int", {"ts_type": type(ts_ns).__name__})
            return None
        if ts_ns is not None and ts_ns <= 0:
            await self._reject(market_id, "non_positive_ts_ns", {"ts_ns": ts_ns})
            return None

        # --- Opportunistic prune + delta compute --------------------------
        self._prune_expired_pauses(now_ns)

        prev = self._last_prob.get(market_id)
        self._touch_last_prob(market_id, prob)
        if prev is None:
            return None

        try:
            delta = prob - prev
        except Exception:  # pragma: no cover - Decimal arithmetic is total
            await self._reject(market_id, "delta_compute_failed", {})
            return None
        # Defense against any pathological Decimal result.
        if not delta.is_finite():
            await self._reject(market_id, "non_finite_delta", {"delta": str(delta)})
            return None

        # --- Detector invocation with containment (Finding #4) ------------
        try:
            anomaly = await self._detector.check(market_id, delta)
        except Exception as exc:
            self._detector_errors += 1
            log.exception(
                "risk.poisoning.detector_error",
                market_id=market_id,
                detector=self._detector.name,
                exc_type=type(exc).__name__,
            )
            # Fail closed: pause the market. Detector failure ≙ loss of
            # signal for this market; conservative default is to block
            # trading on it for the cooldown window.
            until_ns = now_ns + self._pause_sec * 1_000_000_000
            fail_anomaly = Anomaly(
                market_id=market_id,
                detector=self._detector.name,
                score=float("nan"),
                ts_ns=now_ns,
                detail=f"detector_error: {type(exc).__name__}",
                extra={"fail_closed": True, "exception_type": type(exc).__name__},
            )
            self._touch_paused(market_id, _PauseRecord(
                since_ns=now_ns, until_ns=until_ns, anomaly=fail_anomaly,
            ))
            await self._emit(
                Event.make(
                    EventType.POISONING_DETECTOR_ERROR,
                    source=Source.RISK,
                    market_id=market_id,
                    payload={
                        "detector": self._detector.name,
                        "exception_type": type(exc).__name__,
                        "exception_str": str(exc)[:500],
                        "pause_until_ns": until_ns,
                    },
                )
            )
            return None

        if anomaly is None:
            return None

        anomaly_ts_ns = anomaly.ts_ns
        rec = _PauseRecord(
            since_ns=anomaly_ts_ns,
            until_ns=anomaly_ts_ns + self._pause_sec * 1_000_000_000,
            anomaly=anomaly,
        )
        self._touch_paused(market_id, rec)
        log.warning(
            "risk.poisoning.anomaly",
            market_id=market_id,
            detector=anomaly.detector,
            score=anomaly.score,
            detail=anomaly.detail,
        )
        await self._emit(
            Event.make(
                EventType.ANOMALY_DETECTED,
                source=Source.RISK,
                market_id=market_id,
                payload={
                    "detector": anomaly.detector,
                    "score": anomaly.score,
                    "detail": anomaly.detail,
                    "pause_until_ns": rec.until_ns,
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
            "markets_tracked": len(self._last_prob),
            "paused_tracked": len(self._paused),
            "evictions_total": self._evictions_total,
            "prune_cycles": self._prune_cycles,
            "inputs_rejected": self._inputs_rejected,
            "detector_errors": self._detector_errors,
        }
