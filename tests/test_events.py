"""Event enum + envelope."""
from __future__ import annotations

from executor.core.events import Event, EventType, Source


def test_event_type_count_and_uniqueness():
    # Every name in the spec's Decision-4 enumerated list.
    expected = {
        "INTENT_EMITTED", "INTENT_CANCEL_REQUESTED", "STRATEGY_SIGNAL",
        "GATE_CLIPPED", "GATE_REJECTED", "INTENT_ADMITTED", "BASKET_SPLIT",
        "ORDER_PLACED", "ORDER_CANCEL_SENT", "ORDER_REPLACE_SENT",
        "ORDER_ACK", "ORDER_REJECT", "FILL", "ORDER_CANCELLED", "ORDER_EXPIRED",
        "INTENT_COMPLETE", "INTENT_EXPIRED", "BASKET_CANCELLED", "BASKET_ORPHAN",
        "KILL_SWITCH_TOGGLED", "STRATEGY_PAUSED", "STRATEGY_RESUMED",
        "STRATEGY_PAUSE_EXTENDED", "STRATEGY_RESUME_ATTEMPT",
        "CONFIG_RELOADED", "VENUE_HEALTH_TRIPPED", "VENUE_HEALTH_RECOVERED",
        "VENUE_CONNECTION", "EXECUTOR_LIFECYCLE",
        "ACCOUNT_SNAPSHOT", "POSITIONS_SNAPSHOT",
        "ANOMALY_DETECTED", "WARN", "ERROR",
        # Phase 4 — kill switch / telegram lifecycle
        "KILL_COMMAND_RECEIVED", "KILL_STATE_CHANGED",
        # Phase 4.5 — daemon lifecycle + startup pipeline self-check
        "SELF_CHECK_OK", "SELF_CHECK_FAIL", "STATE_SAVED",
    }
    got = {e.value for e in EventType}
    assert got == expected, f"mismatch: missing={expected - got} extra={got - expected}"


def test_event_make_assigns_uuid_and_ts():
    e = Event.make(EventType.WARN, source=Source.EXECUTOR, payload={"msg": "x"})
    assert e.event_id and isinstance(e.event_id, str)
    assert e.ts_ns > 0
    assert e.event_type == EventType.WARN
    assert e.payload == {"msg": "x"}
    assert e.schema_version == 1


def test_source_helpers():
    assert Source.strategy("foo") == "strategy:foo"
    assert Source.venue("kalshi") == "venue:kalshi"
