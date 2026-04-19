"""
Abstract interface for 0g poisoning detectors.

Design goal: the 2.6 risk gate and the orderbook-driven observer wire to this
interface only. Medium (cross-venue divergence) and Broad (synthetic fair value)
implementations drop in via detectors/poisoning/registry.py without touching
gate code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class Anomaly:
    """Emitted when a detector flags a market."""
    market_id: str
    detector: str              # detector name (e.g. "zscore", "cross_venue")
    score: float               # detector-specific severity; |z| for zscore
    ts_ns: int
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class PoisoningDetector(ABC):
    """
    Observers feed (market_id, prob_delta) samples; detector returns an
    Anomaly when its rule fires, else None.

    Implementations MAY keep per-market state (rolling windows, baselines).
    Implementations MUST be safe to call from a single asyncio task.
    """

    name: str = "base"

    @abstractmethod
    async def check(self, market_id: str, prob_delta: Decimal) -> Optional[Anomaly]:
        """Return Anomaly if the newest delta trips the detector, else None."""

    async def reset(self, market_id: str | None = None) -> None:
        """Optional: drop accumulated state. Default no-op."""
        return None

    def snapshot(self) -> dict[str, Any]:
        """Optional: inspection. Default empty."""
        return {}
