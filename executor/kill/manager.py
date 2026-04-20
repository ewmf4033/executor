"""
KillManager — orchestrates SOFT/HARD kill switch + ALL_OR_NONE basket cancel.

Owns:
- KillStateStore (persistent state)
- ref to underlying low-level KillSwitch (gate-facing primitive)
- registered VenueAdapters (for HARD-mode cancel_order across venues)
- registered "in-flight basket" registry: {intent_id: BasketIntent + open_legs}

Flow:
- engage(SOFT, reason, *, panic=False)        -> stop new intents
- engage(HARD, reason, *, panic=False)        -> SOFT + cancel all open orders
- resume(*, force=False)                       -> clears mode if checks pass
- record_basket(intent, open_order_ids_per_leg) -> bookkeeping for ALL_OR_NONE
- on_leg_filled(intent_id, leg_id)             -> mark filled
- on_leg_done(intent_id, leg_id)               -> mark cancelled/expired/rejected

3-strike circuit breaker:
- mark_resume_attempt(healthy: bool)
  - healthy=True   -> reset strikes to 0
  - healthy=False  -> increment strikes; if >= limit, set manual_only

Audit events emitted (via injected publish callback):
- KILL_COMMAND_RECEIVED  — recorded when a command arrives
- KILL_STATE_CHANGED     — when mode/reason transitions
- KILL_SWITCH_TOGGLED    — legacy event from spec; also emitted on engage/release
- BASKET_ORPHAN          — when HARD finds a partially-filled ALL_OR_NONE basket
- BASKET_CANCELLED       — when all legs of an ALL_OR_NONE basket cancelled cleanly

The manager is async-safe: a single internal asyncio.Lock guards every
mutating operation. Reads (is_killed, snapshot) are lock-free.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from ..core.events import Event, EventType, Source
from ..core.intent import Atomicity, BasketIntent
from ..core.logging import get_logger
from ..risk.kill import KillScope, KillSwitch
from .state import KillMode, KillStateSnapshot, KillStateStore


log = get_logger("executor.kill.manager")


Publish = Callable[[Event], Awaitable[None]]


@dataclass
class _BasketRecord:
    intent: BasketIntent
    # leg_id -> set of venue order_ids
    open_orders: dict[str, set[str]] = field(default_factory=dict)
    filled_legs: set[str] = field(default_factory=set)
    cancelled_legs: set[str] = field(default_factory=set)


class _MinimalAdapter:
    """Structural protocol: cancel_order(order_id) -> bool."""

    async def cancel_order(self, order_id: str) -> bool:  # pragma: no cover
        ...


class KillManager:
    def __init__(
        self,
        *,
        store: KillStateStore,
        kill_switch: KillSwitch | None = None,
        publish: Publish | None = None,
        auto_resume_strike_limit: int = 3,
        panic_cooldown_sec: int = 300,
        healthy_window_sec: int = 60,
    ) -> None:
        self._store = store
        self._kill_switch = kill_switch or KillSwitch()
        self._publish = publish
        self._auto_resume_strike_limit = auto_resume_strike_limit
        self._panic_cooldown_sec = panic_cooldown_sec
        self._healthy_window_sec = healthy_window_sec
        self._lock = asyncio.Lock()
        self._snapshot = self._store.load()
        # In-flight baskets we may need to handle on HARD.
        self._baskets: dict[str, _BasketRecord] = {}
        # Adapter registry for HARD-mode cancel.
        self._adapters: dict[str, _MinimalAdapter] = {}
        # If the persisted state shows a non-NONE mode, also reflect into the
        # gate-facing KillSwitch so the very first risk evaluation already sees
        # the global kill.
        if self._snapshot.mode != KillMode.NONE:
            try:
                self._kill_switch.engage(
                    KillScope.GLOBAL, (), self._snapshot.reason or "restored from disk"
                )
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_publish(self, publish: Publish) -> None:
        self._publish = publish

    def register_adapter(self, venue_id: str, adapter: _MinimalAdapter) -> None:
        self._adapters[venue_id] = adapter
        log.info("kill.manager.adapter_registered", venue=venue_id)

    @property
    def kill_switch(self) -> KillSwitch:
        return self._kill_switch

    @property
    def mode(self) -> KillMode:
        return self._snapshot.mode

    def snapshot(self) -> KillStateSnapshot:
        # Defensive copy.
        return KillStateSnapshot(
            mode=self._snapshot.mode,
            reason=self._snapshot.reason,
            engaged_ts_ns=self._snapshot.engaged_ts_ns,
            panic=self._snapshot.panic,
            panic_until_ns=self._snapshot.panic_until_ns,
            resume_strikes=self._snapshot.resume_strikes,
            last_resume_ts_ns=self._snapshot.last_resume_ts_ns,
            manual_only=self._snapshot.manual_only,
            extra=dict(self._snapshot.extra),
        )

    def is_killed(
        self, *, strategy_id: str | None = None, venue: str | None = None
    ) -> tuple[bool, str]:
        # Fast path: SOFT/HARD always block new intents.
        if self._snapshot.mode != KillMode.NONE:
            return True, f"{self._snapshot.mode.value}: {self._snapshot.reason}"
        # Otherwise consult scoped kill registry.
        return self._kill_switch.is_killed(strategy_id=strategy_id, venue=venue)

    # ------------------------------------------------------------------
    # Basket bookkeeping (used by orchestration loop / integration tests)
    # ------------------------------------------------------------------

    def record_basket(
        self,
        intent: BasketIntent,
        *,
        open_orders: dict[str, Iterable[str]],
    ) -> None:
        rec = self._baskets.setdefault(intent.intent_id, _BasketRecord(intent=intent))
        for leg_id, oids in open_orders.items():
            rec.open_orders.setdefault(leg_id, set()).update(oids)

    def mark_leg_filled(self, intent_id: str, leg_id: str) -> None:
        rec = self._baskets.get(intent_id)
        if rec is None:
            return
        rec.filled_legs.add(leg_id)
        # Remove this leg's open orders — they're done.
        rec.open_orders.pop(leg_id, None)

    def mark_leg_done(self, intent_id: str, leg_id: str) -> None:
        rec = self._baskets.get(intent_id)
        if rec is None:
            return
        rec.cancelled_legs.add(leg_id)
        rec.open_orders.pop(leg_id, None)
        # If everything is done, drop the record.
        if not rec.open_orders:
            self._baskets.pop(intent_id, None)

    def open_baskets(self) -> list[BasketIntent]:
        return [r.intent for r in self._baskets.values()]

    # ------------------------------------------------------------------
    # Engage / resume
    # ------------------------------------------------------------------

    async def engage(
        self,
        mode: KillMode,
        reason: str,
        *,
        source: str = "operator",
        panic: bool = False,
        cancel_open_orders: bool = True,
    ) -> KillStateSnapshot:
        """
        Engage SOFT or HARD. HARD additionally cancels all open orders across
        every registered VenueAdapter and emits BASKET_ORPHAN for any partially
        filled ALL_OR_NONE basket.
        """
        if mode == KillMode.NONE:
            raise ValueError("engage requires SOFT or HARD; use resume() to clear")
        if not reason:
            raise ValueError("engage requires a non-empty reason")
        async with self._lock:
            now_ns = time.time_ns()
            prev_mode = self._snapshot.mode
            self._snapshot.mode = mode
            self._snapshot.reason = reason
            self._snapshot.engaged_ts_ns = now_ns
            if panic:
                self._snapshot.panic = True
                self._snapshot.panic_until_ns = (
                    now_ns + self._panic_cooldown_sec * 1_000_000_000
                )
                self._snapshot.manual_only = True
            # Mirror into gate-facing KillSwitch as a GLOBAL.
            try:
                self._kill_switch.engage(KillScope.GLOBAL, (), reason)
            except ValueError:
                pass
            self._store.save(self._snapshot)

            await self._emit_state_changed(prev_mode, mode, reason, source, panic)

            if mode == KillMode.HARD and cancel_open_orders:
                await self._hard_cancel_all()
            return self.snapshot()

    async def resume(
        self,
        *,
        source: str = "operator",
        force: bool = False,
        now_ns: int | None = None,
    ) -> tuple[bool, str]:
        """
        Clear SOFT/HARD mode. Returns (ok, reason_if_blocked).
        Refuses while panic cooldown is active or manual_only is set
        (unless force=True).
        """
        async with self._lock:
            now_ns = now_ns or time.time_ns()
            if self._snapshot.mode == KillMode.NONE:
                return True, ""
            if self._snapshot.panic and self._snapshot.panic_until_ns > now_ns and not force:
                wait_sec = (self._snapshot.panic_until_ns - now_ns) / 1e9
                return False, f"panic cooldown active for {wait_sec:.0f}s more"
            if self._snapshot.manual_only and not force:
                return False, "manual-only mode (force=True required)"
            prev_mode = self._snapshot.mode
            self._snapshot.mode = KillMode.NONE
            self._snapshot.reason = ""
            self._snapshot.panic = False
            self._snapshot.panic_until_ns = 0
            self._snapshot.last_resume_ts_ns = now_ns
            # Drop the GLOBAL entry from the gate-facing kill switch.
            self._kill_switch.release(KillScope.GLOBAL, ())
            self._store.save(self._snapshot)
            await self._emit_state_changed(
                prev_mode, KillMode.NONE, "resume", source, False
            )
            return True, ""

    def mark_resume_health(self, *, healthy: bool, now_ns: int | None = None) -> bool:
        """
        Update strike count after observing a 60s window post-resume.
        Returns True iff manual_only is now set as a result of this call.
        """
        now_ns = now_ns or time.time_ns()
        if healthy:
            if self._snapshot.resume_strikes != 0:
                log.info("kill.manager.strikes.reset")
            self._snapshot.resume_strikes = 0
        else:
            self._snapshot.resume_strikes += 1
            log.warning(
                "kill.manager.strike",
                strike=self._snapshot.resume_strikes,
                limit=self._auto_resume_strike_limit,
            )
            if self._snapshot.resume_strikes >= self._auto_resume_strike_limit:
                self._snapshot.manual_only = True
                log.error("kill.manager.manual_only_pinned")
        self._store.save(self._snapshot)
        return bool(self._snapshot.manual_only)

    def clear_manual_only(self) -> None:
        """Operator override (used by /kill resume --force or admin tools)."""
        self._snapshot.manual_only = False
        self._snapshot.resume_strikes = 0
        self._store.save(self._snapshot)

    # ------------------------------------------------------------------
    # Telegram-side helper: log every received command for audit
    # ------------------------------------------------------------------

    async def emit_command_received(
        self,
        command: str,
        args: str,
        *,
        chat_id: str | int | None = None,
        source: str = Source.TELEGRAM,
    ) -> None:
        if self._publish is None:
            return
        await self._publish(
            Event.make(
                EventType.KILL_COMMAND_RECEIVED,
                source=source,
                payload={"command": command, "args": args, "chat_id": str(chat_id) if chat_id is not None else None},
            )
        )

    # ------------------------------------------------------------------
    # HARD-mode order cancellation across venues + basket orphan handling
    # ------------------------------------------------------------------

    async def _hard_cancel_all(self) -> dict[str, Any]:
        """
        Cancel every open order tracked in self._baskets across registered
        adapters. ALL_OR_NONE baskets with at least one filled leg emit
        BASKET_ORPHAN; otherwise BASKET_CANCELLED is emitted.
        Returns a summary dict for the audit/state-change payload.
        """
        cancelled = 0
        failed = 0
        orphans: list[dict[str, Any]] = []
        completed: list[dict[str, Any]] = []

        # Snapshot the current baskets so concurrent updates don't trip us.
        baskets = list(self._baskets.values())
        for rec in baskets:
            intent = rec.intent
            cancel_results: dict[str, bool] = {}
            for leg in intent.legs:
                oids = list(rec.open_orders.get(leg.leg_id, ()))
                adapter = self._adapters.get(leg.venue)
                for oid in oids:
                    if adapter is None:
                        cancel_results[oid] = False
                        failed += 1
                        continue
                    try:
                        ok = await adapter.cancel_order(oid)
                        cancel_results[oid] = bool(ok)
                        if ok:
                            cancelled += 1
                        else:
                            failed += 1
                    except Exception as exc:
                        log.warning(
                            "kill.manager.cancel_failed",
                            order_id=oid,
                            venue=leg.venue,
                            error=str(exc),
                        )
                        cancel_results[oid] = False
                        failed += 1
                # Either way, this leg has no more pending orders.
                rec.open_orders.pop(leg.leg_id, None)
                if leg.leg_id not in rec.filled_legs:
                    rec.cancelled_legs.add(leg.leg_id)

            # Decide basket-level audit event.
            if intent.atomicity == Atomicity.ALL_OR_NONE and rec.filled_legs:
                payload = {
                    "intent_id": intent.intent_id,
                    "strategy_id": intent.strategy_id,
                    "filled_legs": sorted(rec.filled_legs),
                    "cancelled_legs": sorted(rec.cancelled_legs),
                    "cancel_results": cancel_results,
                    "n_legs": len(intent.legs),
                }
                orphans.append(payload)
                if self._publish is not None:
                    await self._publish(
                        Event.make(
                            EventType.BASKET_ORPHAN,
                            source=Source.KILL,
                            intent_id=intent.intent_id,
                            strategy_id=intent.strategy_id,
                            payload=payload,
                        )
                    )
            else:
                payload = {
                    "intent_id": intent.intent_id,
                    "strategy_id": intent.strategy_id,
                    "cancelled_legs": sorted(rec.cancelled_legs),
                    "cancel_results": cancel_results,
                    "atomicity": intent.atomicity.value,
                }
                completed.append(payload)
                if self._publish is not None:
                    await self._publish(
                        Event.make(
                            EventType.BASKET_CANCELLED,
                            source=Source.KILL,
                            intent_id=intent.intent_id,
                            strategy_id=intent.strategy_id,
                            payload=payload,
                        )
                    )

        # Drop fully-handled baskets.
        self._baskets = {
            iid: rec for iid, rec in self._baskets.items() if rec.open_orders
        }
        summary = {
            "cancelled_orders": cancelled,
            "failed_cancels": failed,
            "orphan_baskets": len(orphans),
            "cancelled_baskets": len(completed),
        }
        log.warning("kill.manager.hard_cancel_done", **summary)
        return summary

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    async def _emit_state_changed(
        self,
        prev: KillMode,
        new: KillMode,
        reason: str,
        source: str,
        panic: bool,
    ) -> None:
        if self._publish is None:
            return
        payload = {
            "from_mode": prev.value,
            "to_mode": new.value,
            "reason": reason,
            "source": source,
            "panic": panic,
            "manual_only": self._snapshot.manual_only,
            "panic_until_ns": self._snapshot.panic_until_ns,
        }
        await self._publish(
            Event.make(
                EventType.KILL_STATE_CHANGED,
                source=Source.KILL,
                payload=payload,
            )
        )
        # Legacy event name from the original spec — keep both.
        await self._publish(
            Event.make(
                EventType.KILL_SWITCH_TOGGLED,
                source=Source.KILL,
                payload=payload,
            )
        )
