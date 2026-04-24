"""
Risk state — SQLite cache + rebuild-on-startup.

Tables:
- positions           : (venue, market_id, outcome_id, size, avg_prob, updated_ts_ns)
- exposures           : cached $ exposure per (venue, market_id, outcome_id)
                        — regenerable from positions at current prices
- daily_pnl           : (date TEXT, strategy_id TEXT, pnl REAL, PRIMARY KEY (date, strategy_id))
                        — strategy_id "" = portfolio-level
- clip_history        : (intent_id TEXT, gate TEXT, original REAL, clipped REAL, ts_ns INT)
- kill_state          : mirror of KillSwitch entries
- adverse_flags       : (strategy_id, market_id, flagged_ts_ns)
- config_hash         : (id INTEGER PRIMARY KEY, hash TEXT, loaded_ts_ns INT)

Rebuild on startup:
1. If risk_state.sqlite exists and opens cleanly, load exposures + daily_pnl + kill + flags + config_hash.
2. Otherwise: delete corrupt file, rebuild exposures from every registered venue
   (get_account + get_positions). Replay audit FILL events since UTC midnight
   into daily_pnl from a provided audit_db_path (if given).
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from ..core.types import Position


log = get_logger("executor.risk.state")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    venue TEXT NOT NULL,
    market_id TEXT NOT NULL,
    outcome_id TEXT NOT NULL,
    size TEXT NOT NULL,
    avg_prob TEXT NOT NULL,
    updated_ts_ns INTEGER NOT NULL,
    PRIMARY KEY (venue, market_id, outcome_id)
);
CREATE TABLE IF NOT EXISTS exposures (
    venue TEXT NOT NULL,
    market_id TEXT NOT NULL,
    outcome_id TEXT NOT NULL,
    dollars TEXT NOT NULL,
    event_id TEXT,
    updated_ts_ns INTEGER NOT NULL,
    PRIMARY KEY (venue, market_id, outcome_id)
);
CREATE TABLE IF NOT EXISTS daily_pnl (
    date TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    pnl TEXT NOT NULL,
    PRIMARY KEY (date, strategy_id)
);
CREATE TABLE IF NOT EXISTS clip_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intent_id TEXT NOT NULL,
    gate TEXT NOT NULL,
    original TEXT NOT NULL,
    clipped TEXT NOT NULL,
    ts_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS kill_state (
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    reason TEXT NOT NULL,
    engaged_ts_ns INTEGER NOT NULL,
    PRIMARY KEY (scope, key)
);
CREATE TABLE IF NOT EXISTS adverse_flags (
    strategy_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    flagged_ts_ns INTEGER NOT NULL,
    PRIMARY KEY (strategy_id, market_id)
);
CREATE TABLE IF NOT EXISTS config_hash (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    hash TEXT NOT NULL,
    loaded_ts_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS strategy_exposures (
    strategy_id TEXT PRIMARY KEY,
    dollars TEXT NOT NULL,
    updated_ts_ns INTEGER NOT NULL
);
-- Phase 4.9 Item 2: durable adverse-selection venue pauses. Pauses stay
-- until explicit operator resume; no automatic TTL on the persisted row.
CREATE TABLE IF NOT EXISTS adverse_selection_pauses (
    venue TEXT PRIMARY KEY,
    paused_at_ns INTEGER NOT NULL,
    reason TEXT,
    source_market_id TEXT
);
-- Phase 4.14b: singleton row tracking operator liveness for Gate 8.5.
-- armed state + timeout + last heartbeat persist across daemon restart,
-- so a crash mid-session does not secretly disarm the dead-man.
CREATE TABLE IF NOT EXISTS operator_liveness (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    armed INTEGER NOT NULL DEFAULT 0,
    armed_ts_ns INTEGER NOT NULL DEFAULT 0,
    timeout_sec INTEGER NOT NULL DEFAULT 0,
    last_heartbeat_ts_ns INTEGER NOT NULL DEFAULT 0,
    armed_by_source TEXT,
    kill_mode_at_arm TEXT,
    disarmed_reason TEXT
);
INSERT OR IGNORE INTO operator_liveness (id) VALUES (1);
"""


@dataclass
class ExposureRecord:
    venue: str
    market_id: str
    outcome_id: str
    dollars: Decimal
    event_id: str | None = None


class RiskStateCorruptInCapitalMode(RuntimeError):
    """Raised by ``RiskState.load(capital_mode=True)`` when the cache is
    corrupt. Phase 4.13.1 Fix #C (GPT-5.5 review #2): under real capital
    we refuse to start trading on reconstructed state; the operator must
    investigate why ``risk_state.sqlite`` became corrupt before resuming.

    Paper mode retains the prior best-effort rebuild-from-venues +
    audit-replay behavior (``capital_mode=False``, the default) so
    observation windows and tests are unaffected.
    """


def utc_midnight_ns(now_ns: int | None = None) -> int:
    now_ns = now_ns or time.time_ns()
    dt = datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc)
    midnight = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    return int(midnight.timestamp() * 1_000_000_000)


def utc_date_str(now_ns: int | None = None) -> str:
    now_ns = now_ns or time.time_ns()
    return datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")


class RiskState:
    """Persistent + in-memory risk state. Always call .load() before use."""

    def __init__(self, *, db_path: str | os.PathLike[str]) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        # In-memory caches (SQLite is the durable backing store).
        self._exposures: dict[tuple[str, str, str], ExposureRecord] = {}
        self._daily_pnl: dict[tuple[str, str], Decimal] = {}   # (date, strategy_id) -> pnl
        self._positions: dict[tuple[str, str, str], Position] = {}
        self._strategy_exposure: dict[str, Decimal] = {}
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(
        self,
        *,
        venues: dict[str, Any] | None = None,
        audit_db_path: str | os.PathLike[str] | None = None,
        capital_mode: bool = False,
    ) -> str:
        """
        Open the SQLite cache. On corruption, delete + rebuild from venues.
        Returns one of: "loaded", "rebuilt_venues", "rebuilt_audit_fallback".

        Phase 4.13.1 Fix #C: when ``capital_mode=True``, corrupt cache is
        NOT auto-rebuilt. Instead ``RiskStateCorruptInCapitalMode`` is
        raised so the daemon refuses to start on reconstructed state.
        Paper mode (``capital_mode=False``, default) preserves the prior
        best-effort rebuild + audit-replay behavior.
        """
        outcome = await asyncio.to_thread(self._try_open)
        if outcome == "ok":
            self._load_caches()
            self._started = True
            log.info("risk.state.loaded", db=str(self._db_path))
            return "loaded"

        # Corrupt / missing path.
        if capital_mode:
            # Fail-closed: do not rebuild from venues, do not replay audit.
            # The operator must investigate the corruption before resuming.
            log.error(
                "risk.state.corrupt_capital_mode_halt",
                reason=outcome,
                db=str(self._db_path),
            )
            raise RiskStateCorruptInCapitalMode(
                f"risk_state.sqlite unreadable under capital_mode "
                f"(reason={outcome}, path={self._db_path}); "
                "refusing to start on reconstructed state — operator must "
                "investigate the corruption before resuming trading"
            )

        log.warning("risk.state.cache_corrupt_rebuilding", reason=outcome, db=str(self._db_path))
        try:
            if self._db_path.exists():
                self._db_path.unlink()
        except OSError:
            pass
        await asyncio.to_thread(self._fresh_open)
        rebuilt_from = "none"
        if venues:
            try:
                rebuilt_from = await self._rebuild_from_venues(venues)
            except Exception as exc:
                log.error("risk.state.venue_rebuild_failed", error=str(exc))
                rebuilt_from = "none"
        if rebuilt_from == "none" and audit_db_path is not None:
            replayed = await asyncio.to_thread(self._replay_audit_fallback, Path(audit_db_path))
            if replayed:
                rebuilt_from = "audit"
        self._load_caches()
        self._started = True
        log.info("risk.state.rebuilt", source=rebuilt_from, db=str(self._db_path))
        return "rebuilt_venues" if rebuilt_from == "venues" else "rebuilt_audit_fallback"

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.commit()
            finally:
                self._conn.close()
            self._conn = None
        self._started = False

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the SQLite handle for co-located stores (Phase 4.14b:
        OperatorLivenessStore). Raises if called before load()."""
        if self._conn is None:
            raise RuntimeError("RiskState.connection accessed before load()")
        return self._conn

    # ------------------------------------------------------------------
    # Open / schema
    # ------------------------------------------------------------------

    def _try_open(self) -> str:
        """Return "ok" if file opens + passes integrity_check, else a reason string."""
        if not self._db_path.exists():
            self._fresh_open()
            return "ok"
        try:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            cur = conn.execute("PRAGMA integrity_check")
            row = cur.fetchone()
            if not row or str(row[0]).lower() != "ok":
                conn.close()
                return f"integrity_check:{row}"
            conn.executescript(SCHEMA_SQL)  # idempotent
            self._conn = conn
            return "ok"
        except sqlite3.DatabaseError as exc:
            return f"db_error:{exc}"

    def _fresh_open(self) -> None:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        self._conn = conn

    def _load_caches(self) -> None:
        assert self._conn is not None
        for row in self._conn.execute(
            "SELECT venue, market_id, outcome_id, dollars, event_id FROM exposures"
        ):
            rec = ExposureRecord(row[0], row[1], row[2], Decimal(row[3]), row[4])
            self._exposures[(row[0], row[1], row[2])] = rec
        for row in self._conn.execute(
            "SELECT date, strategy_id, pnl FROM daily_pnl"
        ):
            self._daily_pnl[(row[0], row[1])] = Decimal(row[2])
        for row in self._conn.execute(
            "SELECT strategy_id, dollars FROM strategy_exposures"
        ):
            self._strategy_exposure[row[0]] = Decimal(row[1])

    # ------------------------------------------------------------------
    # Rebuild from venues
    # ------------------------------------------------------------------

    async def _rebuild_from_venues(self, venues: dict[str, Any]) -> str:
        """venues: dict venue_id -> VenueAdapter. Uses get_account + get_positions."""
        assert self._conn is not None
        any_position = False
        for venue_id, adapter in venues.items():
            try:
                await adapter.get_account()
            except Exception as exc:
                log.warning("risk.state.rebuild.get_account_failed", venue=venue_id, error=str(exc))
            try:
                positions = await adapter.get_positions()
            except NotImplementedError:
                log.warning("risk.state.rebuild.no_get_positions", venue=venue_id)
                continue
            except Exception as exc:
                log.warning("risk.state.rebuild.get_positions_failed", venue=venue_id, error=str(exc))
                continue
            for p in positions:
                any_position = True
                abs_size = abs(p.size)
                price = p.avg_price_prob
                dollars = (abs_size * price).quantize(Decimal("0.01"))
                self._conn.execute(
                    "INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?, ?)",
                    (p.venue, p.market_id, p.outcome_id, str(p.size), str(p.avg_price_prob), int(p.as_of_ts)),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO exposures VALUES (?, ?, ?, ?, ?, ?)",
                    (p.venue, p.market_id, p.outcome_id, str(dollars), None, int(p.as_of_ts)),
                )
        self._conn.commit()
        return "venues" if any_position else "none"

    def _replay_audit_fallback(self, audit_db_path: Path) -> bool:
        """Sum FILL events since 00:00 UTC into daily_pnl[("", "")] as portfolio-level.
        Returns True if at least one event was replayed."""
        if not audit_db_path.exists():
            return False
        try:
            src = sqlite3.connect(str(audit_db_path))
        except sqlite3.DatabaseError:
            return False
        since_ns = utc_midnight_ns()
        n = 0
        pnl = Decimal("0")
        try:
            for row in src.execute(
                "SELECT payload_json, strategy_id FROM events "
                "WHERE event_type='FILL' AND ts_ns >= ?",
                (since_ns,),
            ):
                try:
                    payload = json.loads(row[0])
                except Exception:
                    continue
                # Conservative: just record -fee as pnl delta (realized pnl is Phase 4 attribution).
                fee = Decimal(str(payload.get("fee", "0")))
                pnl -= fee
                n += 1
        finally:
            src.close()
        if n == 0:
            return False
        assert self._conn is not None
        date = utc_date_str()
        self._conn.execute(
            "INSERT OR REPLACE INTO daily_pnl VALUES (?, ?, ?)",
            (date, "", str(pnl)),
        )
        self._conn.commit()
        log.info("risk.state.audit_replay", n=n, pnl=str(pnl))
        return True

    # ------------------------------------------------------------------
    # Exposures
    # ------------------------------------------------------------------

    def exposure(self, venue: str, market_id: str, outcome_id: str) -> Decimal:
        rec = self._exposures.get((venue, market_id, outcome_id))
        return rec.dollars if rec else Decimal("0")

    def exposure_by_market(self, venue: str, market_id: str) -> Decimal:
        return sum(
            (r.dollars for (v, m, _), r in self._exposures.items() if v == venue and m == market_id),
            Decimal("0"),
        )

    def exposure_by_venue(self, venue: str) -> Decimal:
        return sum(
            (r.dollars for (v, _, _), r in self._exposures.items() if v == venue),
            Decimal("0"),
        )

    def exposure_by_event(self, event_id: str) -> Decimal:
        return sum(
            (r.dollars for r in self._exposures.values() if r.event_id == event_id),
            Decimal("0"),
        )

    def total_exposure(self) -> Decimal:
        return sum((r.dollars for r in self._exposures.values()), Decimal("0"))

    def add_exposure(
        self,
        *,
        venue: str,
        market_id: str,
        outcome_id: str,
        dollars: Decimal,
        event_id: str | None = None,
        now_ns: int | None = None,
    ) -> None:
        now_ns = now_ns or time.time_ns()
        key = (venue, market_id, outcome_id)
        prev = self._exposures.get(key)
        new = (prev.dollars if prev else Decimal("0")) + dollars
        rec = ExposureRecord(venue, market_id, outcome_id, new, event_id or (prev.event_id if prev else None))
        self._exposures[key] = rec
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO exposures VALUES (?, ?, ?, ?, ?, ?)",
            (venue, market_id, outcome_id, str(new), rec.event_id, now_ns),
        )
        self._conn.commit()

    def set_event_id(self, venue: str, market_id: str, outcome_id: str, event_id: str | None) -> None:
        """Persist the (venue, market, outcome) -> event_id mapping.

        Phase 4.7 F12: uses INSERT OR REPLACE so a fresh mapping (no row
        yet) is persisted rather than silently dropped by an UPDATE on a
        missing row. Existing dollars are preserved from the in-memory
        record so this never clobbers a booked position.
        """
        key = (venue, market_id, outcome_id)
        rec = self._exposures.get(key)
        now_ns = time.time_ns()
        if rec is None:
            rec = ExposureRecord(venue, market_id, outcome_id, Decimal("0"), event_id)
            self._exposures[key] = rec
        else:
            rec.event_id = event_id
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO exposures VALUES (?, ?, ?, ?, ?, ?)",
            (venue, market_id, outcome_id, str(rec.dollars), event_id, now_ns),
        )
        self._conn.commit()

    def strategy_exposure(self, strategy_id: str) -> Decimal:
        return self._strategy_exposure.get(strategy_id, Decimal("0"))

    def add_strategy_exposure(self, strategy_id: str, dollars: Decimal) -> None:
        new = self._strategy_exposure.get(strategy_id, Decimal("0")) + dollars
        self._strategy_exposure[strategy_id] = new
        self._persist_strategy_exposure(strategy_id, new)

    def set_strategy_exposure(self, strategy_id: str, dollars: Decimal) -> None:
        """Phase 4.7 F13: persists via INSERT OR REPLACE so an allocation
        set on a fresh strategy_id survives restart. Previously UPDATE
        silently dropped writes for strategies without existing rows."""
        self._strategy_exposure[strategy_id] = Decimal(dollars)
        self._persist_strategy_exposure(strategy_id, Decimal(dollars))

    def _persist_strategy_exposure(self, strategy_id: str, dollars: Decimal) -> None:
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO strategy_exposures "
            "(strategy_id, dollars, updated_ts_ns) VALUES (?, ?, ?)",
            (strategy_id, str(dollars), time.time_ns()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Daily PnL
    # ------------------------------------------------------------------

    def daily_pnl(self, strategy_id: str = "", *, now_ns: int | None = None) -> Decimal:
        key = (utc_date_str(now_ns), strategy_id)
        return self._daily_pnl.get(key, Decimal("0"))

    def record_pnl(self, strategy_id: str, pnl: Decimal, *, now_ns: int | None = None) -> None:
        date = utc_date_str(now_ns)
        key = (date, strategy_id)
        new = self._daily_pnl.get(key, Decimal("0")) + pnl
        self._daily_pnl[key] = new
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO daily_pnl VALUES (?, ?, ?)",
            (date, strategy_id, str(new)),
        )
        self._conn.commit()

    def reset_if_new_day(self, *, now_ns: int | None = None) -> None:
        """Drop in-memory daily_pnl entries that aren't today. SQLite keeps full history."""
        today = utc_date_str(now_ns)
        self._daily_pnl = {k: v for k, v in self._daily_pnl.items() if k[0] == today}

    # ------------------------------------------------------------------
    # Clip history
    # ------------------------------------------------------------------

    def record_clip(
        self,
        *,
        intent_id: str,
        gate: str,
        original: Decimal,
        clipped: Decimal,
        now_ns: int | None = None,
    ) -> None:
        now_ns = now_ns or time.time_ns()
        assert self._conn is not None
        self._conn.execute(
            "INSERT INTO clip_history (intent_id, gate, original, clipped, ts_ns) "
            "VALUES (?, ?, ?, ?, ?)",
            (intent_id, gate, str(original), str(clipped), now_ns),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Config hash
    # ------------------------------------------------------------------

    def record_config_hash(self, config_hash: str, *, now_ns: int | None = None) -> None:
        now_ns = now_ns or time.time_ns()
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO config_hash (id, hash, loaded_ts_ns) VALUES (1, ?, ?)",
            (config_hash, now_ns),
        )
        self._conn.commit()

    def current_config_hash(self) -> tuple[str | None, int | None]:
        assert self._conn is not None
        row = self._conn.execute("SELECT hash, loaded_ts_ns FROM config_hash WHERE id=1").fetchone()
        if row is None:
            return None, None
        return row[0], row[1]

    # ------------------------------------------------------------------
    # Phase 4.9 Item 2: durable adverse-selection venue pauses.
    # Kept small + sync because pauses are rare and the detector is the
    # only writer. No TTL — pauses remain until clear_adverse_pause.
    # ------------------------------------------------------------------

    def list_adverse_pauses(self) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT venue, paused_at_ns, reason, source_market_id "
            "FROM adverse_selection_pauses"
        ).fetchall()
        return [
            {
                "venue": r[0],
                "paused_at_ns": int(r[1]),
                "reason": r[2],
                "source_market_id": r[3],
            }
            for r in rows
        ]

    def save_adverse_pause(
        self,
        *,
        venue: str,
        paused_at_ns: int,
        reason: str | None = None,
        source_market_id: str | None = None,
    ) -> None:
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO adverse_selection_pauses "
            "(venue, paused_at_ns, reason, source_market_id) VALUES (?, ?, ?, ?)",
            (venue, int(paused_at_ns), reason, source_market_id),
        )
        self._conn.commit()

    def clear_adverse_pause(self, venue: str) -> None:
        assert self._conn is not None
        self._conn.execute(
            "DELETE FROM adverse_selection_pauses WHERE venue = ?", (venue,)
        )
        self._conn.commit()


# ---------------------------------------------------------------------------
# Phase 4.14b — operator liveness store backing Gate 8.5 (dead-man).
#
# Shares the risk_state.sqlite connection; lives as a small class to keep
# the arm/disarm/heartbeat/status surface co-located. Armed state,
# armed_ts_ns, and last_heartbeat_ts_ns all persist across daemon restart
# — a crash at T=3h with timeout=6h leaves 3h remaining, not a reset.
# ---------------------------------------------------------------------------


@dataclass
class OperatorLivenessSnapshot:
    armed: bool
    armed_ts_ns: int
    timeout_sec: int
    last_heartbeat_ts_ns: int
    armed_by_source: str | None
    kill_mode_at_arm: str | None
    disarmed_reason: str | None


class OperatorLivenessStore:
    """Singleton-row store for operator arm/disarm/heartbeat state.

    The backing table is created by SCHEMA_SQL; the singleton row (id=1)
    is seeded via INSERT OR IGNORE on every connection open so fresh
    DBs have a valid starting row (armed=0).

    Phase 4.14e — atomicity + writer serialization.

    The backing sqlite3 connection is shared with the rest of
    RiskState and is opened with ``check_same_thread=False``, which
    permits cross-thread use at the cost of interleaved
    read/branch/write sequences. Before this change ``heartbeat()``
    performed ``load()`` → branch-on-armed → ``UPDATE``, so a
    ``disarm()`` landing between the load and the update could leave
    the row disarmed while still accepting the heartbeat write. The
    caller would also see ``True`` and treat the heartbeat as
    applied.

    Hardening:
      * ``heartbeat`` is now a single conditional ``UPDATE ...
        WHERE armed = 1``; ``cursor.rowcount`` determines whether
        the write applied (and therefore whether the caller sees
        ``True``).
      * ``_write_lock`` (``threading.RLock``) serializes every
        arm/disarm/heartbeat write + commit sequence, so two writers
        never share the connection cursor mid-transaction.
      * ``load``/``status`` take the same lock: SQLite
        isolation-level is autocommit-per-execute here, but the
        lock keeps reads consistent against concurrent writers
        (no partial ``last_heartbeat_ts_ns`` observed across the
        write boundary).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        # RLock so nested calls on the same thread (e.g. load() from
        # status() while another method holds the lock) don't self-
        # deadlock. Shared singleton lock: only one writer at a time.
        self._write_lock = threading.RLock()
        # Defensive: guarantee singleton row exists even if this store is
        # constructed against a connection opened before SCHEMA_SQL ran.
        with self._write_lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO operator_liveness (id) VALUES (1)"
            )
            self._conn.commit()

    def load(self) -> OperatorLivenessSnapshot:
        with self._write_lock:
            row = self._conn.execute(
                "SELECT armed, armed_ts_ns, timeout_sec, last_heartbeat_ts_ns, "
                "armed_by_source, kill_mode_at_arm, disarmed_reason "
                "FROM operator_liveness WHERE id = 1"
            ).fetchone()
        if row is None:
            # Should be unreachable given INSERT OR IGNORE above, but be
            # explicit rather than returning silently wrong state.
            return OperatorLivenessSnapshot(
                armed=False,
                armed_ts_ns=0,
                timeout_sec=0,
                last_heartbeat_ts_ns=0,
                armed_by_source=None,
                kill_mode_at_arm=None,
                disarmed_reason=None,
            )
        return OperatorLivenessSnapshot(
            armed=bool(row[0]),
            armed_ts_ns=int(row[1]),
            timeout_sec=int(row[2]),
            last_heartbeat_ts_ns=int(row[3]),
            armed_by_source=row[4],
            kill_mode_at_arm=row[5],
            disarmed_reason=row[6],
        )

    def arm(
        self,
        *,
        timeout_sec: int,
        source: str,
        kill_mode: str,
        now_ns: int,
    ) -> None:
        """Engage the dead-man. Records kill_mode at arm time for audit.

        last_heartbeat is set to now_ns so the first 'timeout_sec' window
        starts from arm time; the operator does not need an immediate
        heartbeat after arm.
        """
        with self._write_lock:
            self._conn.execute(
                "UPDATE operator_liveness SET "
                "armed = 1, armed_ts_ns = ?, timeout_sec = ?, "
                "last_heartbeat_ts_ns = ?, armed_by_source = ?, "
                "kill_mode_at_arm = ?, disarmed_reason = NULL "
                "WHERE id = 1",
                (int(now_ns), int(timeout_sec), int(now_ns), source, kill_mode),
            )
            self._conn.commit()

    def disarm(self, *, reason: str, now_ns: int) -> None:
        """Clear armed state, recording reason for audit trail.

        now_ns is accepted for API symmetry and potential future auditing
        of disarm times; it is not currently persisted on the row (the
        OPERATOR_DISARMED audit event carries the timestamp).
        """
        del now_ns  # reserved for future use
        with self._write_lock:
            self._conn.execute(
                "UPDATE operator_liveness SET armed = 0, disarmed_reason = ? "
                "WHERE id = 1",
                (reason,),
            )
            self._conn.commit()

    def heartbeat(self, *, now_ns: int) -> bool:
        """Reset-to-full heartbeat. No-op on disarmed state (returns False).

        Phase 4.14e — atomic conditional write. The armed-state check
        and the last_heartbeat_ts_ns update are now a single
        ``UPDATE ... WHERE armed = 1`` statement. If the row is
        disarmed the ``rowcount`` is zero, ``last_heartbeat_ts_ns``
        is NOT advanced, and the caller sees ``False`` — so a
        concurrent ``disarm()`` landing during a heartbeat cannot
        leave the row with a fresh heartbeat timestamp on a
        disarmed state.

        Returns True if the heartbeat write applied (row was armed),
        False otherwise (no write — does not silently re-arm or
        refresh heartbeat on a disarmed row).
        """
        with self._write_lock:
            cur = self._conn.execute(
                "UPDATE operator_liveness SET last_heartbeat_ts_ns = ? "
                "WHERE id = 1 AND armed = 1",
                (int(now_ns),),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def status(self, *, now_ns: int) -> dict[str, Any]:
        snap = self.load()
        if snap.armed:
            deadline_ns = snap.last_heartbeat_ts_ns + snap.timeout_sec * 1_000_000_000
            seconds_until_stale = (deadline_ns - int(now_ns)) / 1e9
            seconds_since_armed = (int(now_ns) - snap.armed_ts_ns) / 1e9
        else:
            seconds_until_stale = 0.0
            seconds_since_armed = 0.0
        return {
            "armed": snap.armed,
            "armed_ts_ns": snap.armed_ts_ns,
            "timeout_sec": snap.timeout_sec,
            "last_heartbeat_ts_ns": snap.last_heartbeat_ts_ns,
            "armed_by_source": snap.armed_by_source,
            "kill_mode_at_arm": snap.kill_mode_at_arm,
            "disarmed_reason": snap.disarmed_reason,
            "seconds_until_stale": seconds_until_stale,
            "seconds_since_armed": seconds_since_armed,
        }
