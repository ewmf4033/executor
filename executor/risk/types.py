"""
Risk-policy data types.

GateDecision: APPROVE | CLIP | REJECT.
GateResult:   a gate's verdict for one intent (optionally with new leg sizes).
GateCtx:      the per-evaluation state passed to every gate.
RiskVerdict:  the policy-level outcome (final intent, gates_passed, timings).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..core.intent import BasketIntent

if TYPE_CHECKING:
    from .policy import RiskPolicy


class GateDecision(str, Enum):
    APPROVE = "APPROVE"
    CLIP = "CLIP"
    REJECT = "REJECT"


@dataclass(frozen=True, slots=True)
class GateResult:
    decision: GateDecision
    reason: str = ""
    # leg_id -> new size. If None/missing on CLIP, gate wants the whole intent rejected.
    new_leg_sizes: dict[str, Decimal] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def approve(cls, *, metadata: dict[str, Any] | None = None) -> "GateResult":
        return cls(GateDecision.APPROVE, reason="", new_leg_sizes=None,
                   metadata=dict(metadata or {}))

    @classmethod
    def clip(
        cls,
        new_leg_sizes: dict[str, Decimal],
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> "GateResult":
        return cls(GateDecision.CLIP, reason=reason, new_leg_sizes=dict(new_leg_sizes),
                   metadata=dict(metadata or {}))

    @classmethod
    def reject(cls, reason: str, metadata: dict[str, Any] | None = None) -> "GateResult":
        return cls(GateDecision.REJECT, reason=reason, new_leg_sizes=None,
                   metadata=dict(metadata or {}))


@dataclass
class GateCtx:
    """Per-evaluation mutable context. Each gate may read and (carefully) mutate."""
    original_intent: BasketIntent
    current_intent: BasketIntent         # possibly-clipped intent flowing down the pipeline
    policy: "RiskPolicy"
    now_ns: int
    gate_timings_ms: dict[str, float] = field(default_factory=dict)
    clip_history: list[dict[str, Any]] = field(default_factory=list)
    gates_passed: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    admitted: bool
    intent: BasketIntent                 # final intent (may be clipped)
    gates_passed: tuple[str, ...]
    gate_timings_ms: dict[str, float]
    clip_history: tuple[dict[str, Any], ...]
    reject_reason: str | None
    reject_gate: str | None
