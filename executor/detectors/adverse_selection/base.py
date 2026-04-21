"""0e adverse-selection: abstract interface + null (Phase 3) implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdverseSelectionFlag:
    strategy_id: str
    market_id: str
    reason: str
    ts_ns: int


class AdverseSelectionDetector(ABC):
    @abstractmethod
    def is_flagged(self, *, strategy_id: str, market_id: str) -> bool:
        """Legacy market-level flag — kept for callers that still use it."""

    @abstractmethod
    def is_venue_paused(self, venue: str) -> bool:
        """Gate 3 polls this. True => REJECT every leg on `venue`.

        Phase 4.7: adverse-selection pause is enforced at the venue level
        (information asymmetry / latency is venue-wide), not per market.
        Multiple markets on the same venue can trip the underlying metric;
        the venue stays paused until explicit resume.
        """


class NullAdverseSelectionDetector(AdverseSelectionDetector):
    """Tests-only default: never flags, never pauses. Must be passed
    explicitly to RiskPolicy — the Phase 4.7 policy no longer defaults
    to this detector, so production can't accidentally run without a
    real one wired."""

    def is_flagged(self, *, strategy_id: str, market_id: str) -> bool:
        return False

    def is_venue_paused(self, venue: str) -> bool:
        return False
