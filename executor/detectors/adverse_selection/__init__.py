"""
0e adverse-selection detector.

Phase 3 shipped the AdverseSelectionDetector interface + Null implementation.
Phase 4 adds WindowAdverseSelectionDetector — sliding-window fill-vs-post-fill
detection that pauses venues showing systematic adverse selection.
"""
from .base import AdverseSelectionDetector, AdverseSelectionFlag, NullAdverseSelectionDetector
from .window import WindowAdverseSelectionDetector

__all__ = [
    "AdverseSelectionDetector",
    "AdverseSelectionFlag",
    "NullAdverseSelectionDetector",
    "WindowAdverseSelectionDetector",
]
