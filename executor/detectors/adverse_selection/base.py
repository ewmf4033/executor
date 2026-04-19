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
        """Gate 3 polls this. True => REJECT."""


class NullAdverseSelectionDetector(AdverseSelectionDetector):
    """Phase 3 default: never flags. Phase 4 replaces with real implementation."""

    def is_flagged(self, *, strategy_id: str, market_id: str) -> bool:
        return False
