# DECISIONS

Deviations from `/root/trading-wiki/specs/0d-executor.md` (version 2026-04-19). Record here
only when implementation reality forces a change. Style preferences are not valid reasons.

Format per entry:
```
## YYYY-MM-DD — <title>
**Spec reference:** <section/decision>
**Deviation:** <what changed>
**Forced by:** <reason>
**Impact:** <what else this touches>
```

---

## 2026-04-19 — Event type count
**Spec reference:** Decision 4, "29 event types"
**Deviation:** Implemented 34 event types, not 29.
**Forced by:** The spec prose says "29 event types" but the enumerated list
across the groups (Strategy→Executor 3, Risk 4, Executor→Venue 3,
Venue→Executor 5, Executor internal 4, State 10, Observations 2,
Diagnostics 3) sums to 34. Every named event type is implemented verbatim;
the "29" number in the overview prose appears to be a stale count.
**Impact:** `executor/core/events.py::EventType` has 34 members with an
`assert len(EventType) == 34` guard. If the authoritative count is meant
to be 29, the spec must name which 5 enumerated types to drop; flag on
spec re-review.
