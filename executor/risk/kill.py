"""
Minimal kill-switch primitive used by Gate 2 in Phase 3.

The full kill-switch (HARD, basket orphan semantics, Telegram) is Phase 4.
Here we provide just enough: scoped engage/release + is_killed(scope) query,
so Gate 2 has something real to poll and unit tests can exercise it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from ..core.logging import get_logger


log = get_logger("executor.risk.kill")


class KillScope(str, Enum):
    GLOBAL = "GLOBAL"
    STRATEGY = "STRATEGY"
    VENUE = "VENUE"
    STRATEGY_VENUE = "STRATEGY_VENUE"


@dataclass(frozen=True, slots=True)
class KillEntry:
    scope: KillScope
    key: tuple[str, ...]           # () for GLOBAL, (sid,), (venue,), or (sid, venue)
    reason: str
    engaged_ts_ns: int


class KillSwitch:
    """In-memory kill registry. Scopes stack: GLOBAL overrides everything."""

    def __init__(self) -> None:
        self._entries: dict[tuple[KillScope, tuple[str, ...]], KillEntry] = {}

    def engage(self, scope: KillScope, key: Iterable[str], reason: str) -> None:
        if not reason:
            raise ValueError("kill engage requires a reason")
        k = tuple(key)
        self._entries[(scope, k)] = KillEntry(scope, k, reason, time.time_ns())
        log.warning("risk.kill.engage", scope=scope.value, key=list(k), reason=reason)

    def release(self, scope: KillScope, key: Iterable[str]) -> bool:
        k = tuple(key)
        removed = self._entries.pop((scope, k), None) is not None
        if removed:
            log.info("risk.kill.release", scope=scope.value, key=list(k))
        return removed

    def is_killed(
        self,
        *,
        strategy_id: str | None = None,
        venue: str | None = None,
    ) -> tuple[bool, str]:
        """Returns (killed, reason). First-match wins in GLOBAL > STRATEGY_VENUE > STRATEGY > VENUE order."""
        hit = self._entries.get((KillScope.GLOBAL, ()))
        if hit:
            return True, f"GLOBAL: {hit.reason}"
        if strategy_id and venue:
            hit = self._entries.get((KillScope.STRATEGY_VENUE, (strategy_id, venue)))
            if hit:
                return True, f"STRATEGY_VENUE {strategy_id}/{venue}: {hit.reason}"
        if strategy_id:
            hit = self._entries.get((KillScope.STRATEGY, (strategy_id,)))
            if hit:
                return True, f"STRATEGY {strategy_id}: {hit.reason}"
        if venue:
            hit = self._entries.get((KillScope.VENUE, (venue,)))
            if hit:
                return True, f"VENUE {venue}: {hit.reason}"
        return False, ""

    def entries(self) -> list[KillEntry]:
        return list(self._entries.values())
