"""
Audit event model — the 29 event types locked in Decision 4.

Spec: /root/trading-wiki/specs/0d-executor.md

Every event has a time-sortable UUID v7 event_id, a nanosecond timestamp,
a source ("strategy:<id>", "executor", "venue:<id>", "risk", "kill", "telegram"),
optional correlation keys (intent_id, leg_id, venue, market_id, strategy_id),
a JSON payload, and a schema_version.

Schema evolution rule: never mutate an existing event type. New behavior =
new event type (e.g. INTENT_EMITTED_V2). Readers tolerate missing fields.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from uuid6 import uuid7


# ---------------------------------------------------------------------------
# 29 event types, grouped per spec.
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    # Strategy -> Executor (3)
    INTENT_EMITTED = "INTENT_EMITTED"
    INTENT_CANCEL_REQUESTED = "INTENT_CANCEL_REQUESTED"
    STRATEGY_SIGNAL = "STRATEGY_SIGNAL"

    # Risk decisions (4)
    GATE_CLIPPED = "GATE_CLIPPED"
    GATE_REJECTED = "GATE_REJECTED"
    INTENT_ADMITTED = "INTENT_ADMITTED"
    BASKET_SPLIT = "BASKET_SPLIT"

    # Executor -> Venue (3)
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_CANCEL_SENT = "ORDER_CANCEL_SENT"
    ORDER_REPLACE_SENT = "ORDER_REPLACE_SENT"

    # Venue -> Executor (5)
    ORDER_ACK = "ORDER_ACK"
    ORDER_REJECT = "ORDER_REJECT"
    FILL = "FILL"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_EXPIRED = "ORDER_EXPIRED"

    # Executor internal (4)
    INTENT_COMPLETE = "INTENT_COMPLETE"
    INTENT_EXPIRED = "INTENT_EXPIRED"
    BASKET_CANCELLED = "BASKET_CANCELLED"
    BASKET_ORPHAN = "BASKET_ORPHAN"

    # State (7)
    KILL_SWITCH_TOGGLED = "KILL_SWITCH_TOGGLED"
    STRATEGY_PAUSED = "STRATEGY_PAUSED"
    STRATEGY_RESUMED = "STRATEGY_RESUMED"
    STRATEGY_PAUSE_EXTENDED = "STRATEGY_PAUSE_EXTENDED"
    STRATEGY_RESUME_ATTEMPT = "STRATEGY_RESUME_ATTEMPT"
    CONFIG_RELOADED = "CONFIG_RELOADED"
    VENUE_HEALTH_TRIPPED = "VENUE_HEALTH_TRIPPED"
    VENUE_HEALTH_RECOVERED = "VENUE_HEALTH_RECOVERED"
    VENUE_CONNECTION = "VENUE_CONNECTION"
    EXECUTOR_LIFECYCLE = "EXECUTOR_LIFECYCLE"
    # Phase 4: telegram + kill switch lifecycle
    KILL_COMMAND_RECEIVED = "KILL_COMMAND_RECEIVED"
    KILL_STATE_CHANGED = "KILL_STATE_CHANGED"

    # Observations (2)
    ACCOUNT_SNAPSHOT = "ACCOUNT_SNAPSHOT"
    POSITIONS_SNAPSHOT = "POSITIONS_SNAPSHOT"

    # Diagnostics (3)
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    WARN = "WARN"
    ERROR = "ERROR"

    # Phase 4.5 — daemon lifecycle + startup self-check. Added per the
    # "new behavior = new event type" rule so the pipeline-verification
    # signal is first-class in the audit log (not buried in a WARN.kind).
    SELF_CHECK_OK = "SELF_CHECK_OK"
    SELF_CHECK_FAIL = "SELF_CHECK_FAIL"
    STATE_SAVED = "STATE_SAVED"

    # Phase 4.11 — post-admission kill recheck (Review 9 #3). Distinct from
    # BASKET_CANCELLED so Phase 5 alerting can separately track orders that
    # were killed after risk admission but before venue emission.
    ORDER_CANCELLED_PRE_SEND = "ORDER_CANCELLED_PRE_SEND"

    # Phase 4.12 — 0g hardening: reject malformed inputs at poisoning boundary.
    POISONING_INPUT_REJECTED = "POISONING_INPUT_REJECTED"
    # Phase 4.12 — 0g hardening: detector raised; tracker fail-closed the market.
    POISONING_DETECTOR_ERROR = "POISONING_DETECTOR_ERROR"


# Decision 4 prose reads "29 event types" but the enumerated list across
# groups (3+4+3+5+4+10+2+3) totals 34. We implement every name the spec
# explicitly lists. See DECISIONS.md "2026-04-19 — Event type count" for
# the reconciliation. Phase 4 added KILL_COMMAND_RECEIVED + KILL_STATE_CHANGED
# (telegram bot + kill-switch lifecycle). Phase 4.5 added SELF_CHECK_OK,
# SELF_CHECK_FAIL, STATE_SAVED (daemon lifecycle + pipeline self-check).
# Phase 4.11 added ORDER_CANCELLED_PRE_SEND (Review 9 #3: kill switch
# between risk admission and venue emit). Phase 4.12 added
# POISONING_INPUT_REJECTED + POISONING_DETECTOR_ERROR (0g hardening:
# input validation + detector exception fail-closed).
# Guard against silent drift:
assert len(EventType) == 42, f"EventType count drift: {len(EventType)}"


# ---------------------------------------------------------------------------
# Source tags — conventional strings for event.source. Not enforced, but keep
# the set small so the audit log is queryable.
# ---------------------------------------------------------------------------


class Source:
    EXECUTOR = "executor"
    RISK = "risk"
    KILL = "kill"
    TELEGRAM = "telegram"

    @staticmethod
    def strategy(strategy_id: str) -> str:
        return f"strategy:{strategy_id}"

    @staticmethod
    def venue(venue: str) -> str:
        return f"venue:{venue}"


# ---------------------------------------------------------------------------
# Event envelope.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    ts_ns: int
    event_type: EventType
    source: str
    payload: dict[str, Any]
    intent_id: str | None = None
    leg_id: str | None = None
    venue: str | None = None
    market_id: str | None = None
    strategy_id: str | None = None
    schema_version: int = 1

    @classmethod
    def make(
        cls,
        event_type: EventType,
        source: str,
        payload: dict[str, Any] | None = None,
        *,
        intent_id: str | None = None,
        leg_id: str | None = None,
        venue: str | None = None,
        market_id: str | None = None,
        strategy_id: str | None = None,
        ts_ns: int | None = None,
        schema_version: int = 1,
    ) -> "Event":
        return cls(
            event_id=str(uuid7()),
            ts_ns=ts_ns if ts_ns is not None else time.time_ns(),
            event_type=event_type,
            source=source,
            payload=dict(payload or {}),
            intent_id=intent_id,
            leg_id=leg_id,
            venue=venue,
            market_id=market_id,
            strategy_id=strategy_id,
            schema_version=schema_version,
        )
