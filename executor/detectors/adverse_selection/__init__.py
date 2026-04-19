"""
0e adverse-selection detector — interface-only stub for Phase 3.

Phase 4 will implement the real fill-vs-post-fill logic. Gate 3 uses this
stub shape so the call-site code lands once and the real detector drops
in without gate changes.
"""
from .base import AdverseSelectionDetector, AdverseSelectionFlag, NullAdverseSelectionDetector

__all__ = [
    "AdverseSelectionDetector",
    "AdverseSelectionFlag",
    "NullAdverseSelectionDetector",
]
