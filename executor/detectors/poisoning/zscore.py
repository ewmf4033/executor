"""
ZScoreDetector — rolling-window |z|>threshold on prob deltas.

Per spec: "Rolling 1-hour window of prob deltas per market. Flag if latest
delta abs z-score > 5."

Requires `min_samples` observations in the window before any flag. Deltas
more than window_sec old are evicted at each observation.

Phase 4.12 hardening: per-market _Window entries are LRU-capped at
MAX_MARKETS to defend against adversarial feeds allocating unbounded
market_ids. Oldest-by-last-access is evicted when the cap is hit.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Deque, Optional

from .base import Anomaly, PoisoningDetector


# Aligned with tracker.MAX_MARKETS. See tracker.py for rationale.
MAX_MARKETS = 10_000


@dataclass
class _Window:
    samples: Deque[tuple[int, float]]      # (ts_ns, delta)


class ZScoreDetector(PoisoningDetector):
    name = "zscore"

    def __init__(
        self,
        *,
        window_sec: int = 3600,
        z_threshold: float = 5.0,
        min_samples: int = 20,
    ) -> None:
        self.window_sec = window_sec
        self.z_threshold = z_threshold
        self.min_samples = min_samples
        # OrderedDict gives O(1) LRU eviction via move_to_end +
        # popitem(last=False).
        self._windows: "OrderedDict[str, _Window]" = OrderedDict()
        self._evictions_total = 0

    def _touch(self, market_id: str) -> _Window:
        """Return window for market_id, updating LRU order. Evict oldest
        if we would exceed MAX_MARKETS on a new insert.
        """
        w = self._windows.get(market_id)
        if w is not None:
            self._windows.move_to_end(market_id)
            return w
        if len(self._windows) >= MAX_MARKETS:
            self._windows.popitem(last=False)
            self._evictions_total += 1
        w = _Window(samples=deque())
        self._windows[market_id] = w
        return w

    async def check(self, market_id: str, prob_delta: Decimal) -> Optional[Anomaly]:
        now_ns = time.time_ns()
        w = self._touch(market_id)
        cutoff = now_ns - self.window_sec * 1_000_000_000
        while w.samples and w.samples[0][0] < cutoff:
            w.samples.popleft()
        delta_f = float(prob_delta)
        w.samples.append((now_ns, delta_f))
        if len(w.samples) < self.min_samples:
            return None
        values = [s[1] for s in w.samples]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        if std == 0.0:
            return None
        z = (delta_f - mean) / std
        if abs(z) <= self.z_threshold:
            return None
        return Anomaly(
            market_id=market_id,
            detector=self.name,
            score=float(z),
            ts_ns=now_ns,
            detail=f"|z|={abs(z):.2f} > {self.z_threshold}",
            extra={"mean": mean, "std": std, "delta": delta_f, "n": len(values)},
        )

    async def reset(self, market_id: str | None = None) -> None:
        if market_id is None:
            self._windows.clear()
        else:
            self._windows.pop(market_id, None)

    def snapshot(self) -> dict[str, Any]:
        return {
            "markets_tracked": len(self._windows),
            "evictions_total": self._evictions_total,
            "per_market": {
                m: {"n_samples": len(w.samples)} for m, w in self._windows.items()
            },
        }
