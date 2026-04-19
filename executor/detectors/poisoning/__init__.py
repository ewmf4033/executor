"""0g — data-poisoning detectors.

Tight scope: single pluggable interface (PoisoningDetector) + one default
implementation (ZScoreDetector). Medium (cross-venue) and Broad (synthetic
fair value) implementations register through detectors/poisoning/registry.py
without touching the gate wiring.
"""
from .base import Anomaly, PoisoningDetector
from .zscore import ZScoreDetector
from .registry import register_detector, build_detector, list_detectors
from .tracker import PoisoningTracker

__all__ = [
    "Anomaly",
    "PoisoningDetector",
    "ZScoreDetector",
    "register_detector",
    "build_detector",
    "list_detectors",
    "PoisoningTracker",
]
