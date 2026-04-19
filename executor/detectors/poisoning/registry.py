"""
Single registration point for poisoning-detector implementations.

Medium (cross-venue divergence) and Broad (synthetic fair-value) detectors
register factories here, then ConfigManager picks one by name from
risk.yaml::poisoning.detector without touching gate code.
"""
from __future__ import annotations

from typing import Any, Callable

from .base import PoisoningDetector
from .zscore import ZScoreDetector


DetectorFactory = Callable[..., PoisoningDetector]
_REGISTRY: dict[str, DetectorFactory] = {}


def register_detector(name: str, factory: DetectorFactory) -> None:
    if name in _REGISTRY:
        raise ValueError(f"detector {name!r} already registered")
    _REGISTRY[name] = factory


def build_detector(name: str, **kwargs: Any) -> PoisoningDetector:
    if name not in _REGISTRY:
        raise ValueError(f"unknown detector {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_detectors() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in detectors
# ---------------------------------------------------------------------------

register_detector(
    "zscore",
    lambda window_sec=3600, z_threshold=5.0, min_samples=20, **_:
        ZScoreDetector(window_sec=window_sec, z_threshold=z_threshold, min_samples=min_samples),
)
