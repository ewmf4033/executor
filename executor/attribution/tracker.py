"""
AttributionTracker — 0f execution attribution.

For each filled order leg we record:
  intent_price   = strategy's price_limit (what the strategy wanted)
  decision_price = mid at intent emission
  arrival_price  = mid at order placement
  fill_price     = actual fill price
  exit_price     = mid at fill_ts + exit_horizon_sec (default 300s)

Slippage breakdown (positive = adverse for the strategy):
  strategy_edge       = decision - arrival   (adverse if mid moved against us
                                              between strategy decision and
                                              order placement)
  execution_cost      = arrival - fill       (adverse if we paid worse than
                                              the mid at arrival)
  short_term_alpha    = signed(side, fill - exit)
                        positive = adverse to strategy (mid moved against fill).
                        For PnL gating, see record_pnl plumbing in settle_due
                        which uses true realized PnL, not short_term_alpha.

Sign convention for BUY (long YES): all "adverse" terms above are computed
with the assumption that lower price = good when buying back, and we negate
for SELL legs at quantize-time.

Records persist to attribution table in the provided SQLite path. The
GET /attribution/summary endpoint (server.py) returns aggregate slippage
per strategy_id over a configurable lookback.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..core.events import Event, EventType, Source
from ..core.logging import get_logger
from ..core.types import Side

if False:  # TYPE_CHECKING — avoid circular import at runtime
    from ..risk.state import RiskState


log = get_logger("executor.attribution")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS attribution (
    fill_id TEXT PRIMARY KEY,
    intent_id TEXT NOT NULL,
    leg_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    market_id TEXT NOT NULL,
    side TEXT NOT NULL,
    size TEXT NOT NULL,
    intent_price TEXT,
    decision_price TEXT,
    arrival_price TEXT,
    fill_price TEXT NOT NULL,
    exit_price TEXT,
    strategy_edge TEXT,
    execution_cost TEXT,
    short_term_alpha TEXT,
    fee TEXT,
    cost_basis_dollars TEXT,
    venue_fee_bps TEXT,
    fill_ts_ns INTEGER NOT NULL,
    settled_ts_ns INTEGER NOT NULL,
    extra_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_attr_strat_ts ON attribution(strategy_id, fill_ts_ns);
CREATE INDEX IF NOT EXISTS idx_attr_intent ON attribution(intent_id);
"""


@dataclass
class AttributionRecord:
    fill_id: str
    intent_id: str
    leg_id: str
    strategy_id: str
    venue: str
    market_id: str
    side: Side
    size: Decimal
    fill_price: Decimal
    fill_ts_ns: int
    intent_price: Decimal | None = None
    decision_price: Decimal | None = None
    arrival_price: Decimal | None = None
    exit_price: Decimal | None = None
    strategy_edge: Decimal | None = None
    execution_cost: Decimal | None = None
    short_term_alpha: Decimal | None = None
    fee: Decimal | None = None
    cost_basis_dollars: Decimal | None = None
    venue_fee_bps: Decimal | None = None
    settled_ts_ns: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if isinstance(v, Decimal):
                d[k] = str(v)
            elif isinstance(v, Side):
                d[k] = v.value
            else:
                d[k] = v
        return d


@dataclass
class _PendingFill:
    """In-flight: we have the fill but exit_price horizon hasn't elapsed."""
    record: AttributionRecord
    exit_horizon_ns: int


class AttributionTracker:
    """
    Lifecycle:
      tracker = AttributionTracker(db_path=..., exit_horizon_sec=300)
      tracker.note_intent(intent_id, decision_price)
      tracker.note_arrival(intent_id, leg_id, arrival_price)
      tracker.on_fill(fill, intent_id, leg_id, strategy_id, intent_price=..., fee=...)
      tracker.update_mid(market_id, mid)   # called on every quote tick
      # exits are settled in tracker.settle_due(now_ns) — usually in a periodic task

    All methods are sync; SQLite is opened in WAL mode and shared via
    check_same_thread=False, so the orchestration loop can fan-out work
    without ceremony.
    """

    def __init__(
        self,
        *,
        db_path: str | Path,
        exit_horizon_sec: int = 300,
        publish=None,
        risk_state: "RiskState | None" = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        self._exit_horizon_sec = exit_horizon_sec
        self._publish = publish
        # Phase 4.11 (Review 8 finding 0c-3): settlement realized-PnL delta
        # is written to RiskState so gate_13 (daily_loss) actually has a
        # non-zero counter to evaluate. Optional at ctor time so unit
        # tests that don't need PnL integration remain lightweight.
        self._risk_state = risk_state
        # Decision/arrival prices we've recorded ahead of fills.
        self._decision: dict[str, Decimal] = {}             # intent_id -> mid
        self._arrival: dict[tuple[str, str], Decimal] = {}  # (intent_id, leg_id) -> mid
        # Phase 4.9 Item 3: per-intent insertion ts for the max-age sweeper.
        self._decision_ts_ns: dict[str, int] = {}
        self._latest_mid: dict[tuple[str, str], Decimal] = {}  # (venue, market_id) -> mid
        self._pending: list[_PendingFill] = []
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_publish(self, publish) -> None:
        self._publish = publish

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()

    # ------------------------------------------------------------------
    # Observation API
    # ------------------------------------------------------------------

    def note_decision(self, intent_id: str, decision_price: Decimal) -> None:
        self._decision[intent_id] = Decimal(str(decision_price))
        self._decision_ts_ns[intent_id] = time.time_ns()

    def note_arrival(self, intent_id: str, leg_id: str, arrival_price: Decimal) -> None:
        self._arrival[(intent_id, leg_id)] = Decimal(str(arrival_price))
        # Treat arrival as extending the intent's "active" window.
        self._decision_ts_ns.setdefault(intent_id, time.time_ns())

    # ------------------------------------------------------------------
    # Phase 4.9 Item 3: pruning — prevent unbounded growth of the decision
    # / arrival in-memory caches. Called from the orchestrator whenever an
    # intent reaches a terminal state (FILLED / REJECTED_FINAL / EXPIRED).
    # ------------------------------------------------------------------

    def prune_intent(self, intent_id: str) -> int:
        """Remove all cached decision + arrival prices for this intent.
        Returns the number of entries removed (useful for tests). Safe to
        call with an unknown intent_id — it's a no-op then."""
        removed = 0
        if self._decision.pop(intent_id, None) is not None:
            removed += 1
        self._decision_ts_ns.pop(intent_id, None)
        # Remove every (intent_id, leg_id) key for this intent.
        dead = [k for k in self._arrival if k[0] == intent_id]
        for k in dead:
            self._arrival.pop(k, None)
            removed += 1
        return removed

    def prune_older_than(
        self, *, max_age_sec: float, now_ns: int | None = None
    ) -> int:
        """Sweep intents whose decision/arrival records are older than
        `max_age_sec`. Used as a safety net for the rare case where a
        terminal-state notification is dropped (bus crash, orchestrator
        bug). Returns count of intents pruned."""
        now_ns = now_ns or time.time_ns()
        cutoff_ns = now_ns - int(max_age_sec * 1_000_000_000)
        stale = [iid for iid, ts in self._decision_ts_ns.items() if ts < cutoff_ns]
        total = 0
        for iid in stale:
            total += self.prune_intent(iid)
        # Also sweep orphaned (intent, leg) pairs whose intent has no
        # decision_ts anchor — these can only exist if note_arrival was
        # called without note_decision, which is not expected, but guard.
        orphan_intents = {
            k[0] for k in self._arrival if k[0] not in self._decision_ts_ns
        }
        for iid in orphan_intents:
            # Best-effort: we don't know ts for these, so sweep
            # unconditionally when their owning intent is gone.
            self.prune_intent(iid)
        return total

    def update_mid(self, venue: str, market_id: str, mid: Decimal) -> None:
        self._latest_mid[(venue, market_id)] = Decimal(str(mid))

    # ------------------------------------------------------------------
    # On fill
    # ------------------------------------------------------------------

    def on_fill(
        self,
        *,
        fill_id: str,
        order_id: str,
        intent_id: str,
        leg_id: str,
        strategy_id: str,
        venue: str,
        market_id: str,
        side: Side,
        size: Decimal,
        fill_price: Decimal,
        fill_ts_ns: int,
        intent_price: Decimal | None,
        fee: Decimal | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AttributionRecord:
        decision = self._decision.get(intent_id)
        arrival = self._arrival.get((intent_id, leg_id))
        rec = AttributionRecord(
            fill_id=fill_id,
            intent_id=intent_id,
            leg_id=leg_id,
            strategy_id=strategy_id,
            venue=venue,
            market_id=market_id,
            side=side,
            size=Decimal(str(size)),
            fill_price=Decimal(str(fill_price)),
            fill_ts_ns=int(fill_ts_ns),
            intent_price=Decimal(str(intent_price)) if intent_price is not None else None,
            decision_price=decision,
            arrival_price=arrival,
            exit_price=None,
            fee=Decimal(str(fee)) if fee is not None else None,
            extra=dict(extra or {}),
        )
        # Slippage components now (exit-related is filled in at settle).
        if decision is not None and arrival is not None:
            rec.strategy_edge = _signed(rec.side, decision - arrival)
        if arrival is not None:
            rec.execution_cost = _signed(rec.side, arrival - rec.fill_price)
        # Cost basis and venue fee.
        notional = rec.size * rec.fill_price
        if rec.fee is not None:
            if rec.side == Side.BUY:
                rec.cost_basis_dollars = notional + rec.fee
            else:
                rec.cost_basis_dollars = notional - rec.fee
            if notional > 0:
                rec.venue_fee_bps = (rec.fee / notional) * Decimal("10000")
        else:
            rec.cost_basis_dollars = notional
        # Persist a partial row (no exit yet); we update at settlement.
        self._upsert(rec)
        # Schedule settlement.
        exit_ns = rec.fill_ts_ns + self._exit_horizon_sec * 1_000_000_000
        self._pending.append(_PendingFill(record=rec, exit_horizon_ns=exit_ns))
        return rec

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    async def settle_due(self, now_ns: int | None = None) -> list[AttributionRecord]:
        """Finalize any pending fill whose exit horizon has elapsed."""
        now_ns = now_ns or time.time_ns()
        settled: list[AttributionRecord] = []
        keep: list[_PendingFill] = []
        for p in self._pending:
            if p.exit_horizon_ns > now_ns:
                keep.append(p)
                continue
            mid = self._latest_mid.get((p.record.venue, p.record.market_id))
            if mid is not None:
                p.record.exit_price = Decimal(str(mid))
                p.record.short_term_alpha = _signed(
                    p.record.side, p.record.fill_price - p.record.exit_price
                )
            p.record.settled_ts_ns = now_ns
            self._upsert(p.record)
            # Phase 4.11.2: see record_pnl call below for true PnL computation.
            # short_term_alpha is preserved on the record for attribution
            # analysis but is NOT used as a PnL feed (adverse-positive
            # convention would invert gate_13/daily_loss economics).
            #
            # Phase 4.11.2 (Codex Review 10): replace alpha-derived PnL with
            # true realized PnL at exit horizon. short_term_alpha uses
            # adverse-positive convention (see _signed() docstring), making
            # it unsuitable as a direct PnL feed to record_pnl(). True PnL
            # is direction-aware:
            #   BUY:  pnl = (exit_price - fill_price) * size  (price up = profit)
            #   SELL: pnl = (fill_price - exit_price) * size  (price down = profit)
            # We use a sign multiplier for compactness.
            #
            # Phase 4.13.2 (GPT-5.5 architectural review #2, 2026-04-23):
            # subtract fees so gate_13 (daily_loss) sees NET-of-fees PnL.
            # Kalshi fees are always a cost paid by the trader (stored as a
            # positive Decimal on AttributionRecord), so net = gross - fee
            # regardless of side. At small T1 trade sizes, fee rounding can
            # flip a gross-profitable strategy to net-negative, so gate_13
            # must be fed the net number.
            if (
                self._risk_state is not None
                and p.record.exit_price is not None
                and p.record.fill_price is not None
            ):
                try:
                    side_sign = Decimal("1") if p.record.side == Side.BUY else Decimal("-1")
                    gross_pnl = side_sign * (p.record.exit_price - p.record.fill_price) * p.record.size
                    # Fee is a cost (always subtracted); None → 0 for paper
                    # fills and any path that hasn't populated fee yet.
                    fee = p.record.fee if p.record.fee is not None else Decimal("0")
                    pnl_delta = gross_pnl - fee
                    self._risk_state.record_pnl(
                        p.record.strategy_id,
                        pnl_delta,
                        now_ns=now_ns,
                    )
                except Exception as exc:  # pragma: no cover
                    log.warning(
                        "attribution.record_pnl_failed",
                        error=str(exc),
                        strategy_id=p.record.strategy_id,
                        fill_id=p.record.fill_id,
                    )
            settled.append(p.record)
            if self._publish is not None:
                try:
                    await self._publish(
                        Event.make(
                            # Reuse WARN as a generic notice so we don't add
                            # yet another EventType. Payload kind tags it.
                            EventType.WARN,
                            source=Source.EXECUTOR,
                            intent_id=p.record.intent_id,
                            leg_id=p.record.leg_id,
                            venue=p.record.venue,
                            market_id=p.record.market_id,
                            strategy_id=p.record.strategy_id,
                            payload={
                                "kind": "ATTRIBUTION_SETTLED",
                                "record": p.record.to_dict(),
                            },
                        )
                    )
                except Exception as exc:  # pragma: no cover
                    log.warning("attribution.emit_failed", error=str(exc))
        self._pending = keep
        return settled

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def summary(
        self, *, since_ns: int = 0, strategy_id: str | None = None
    ) -> dict[str, Any]:
        sql = (
            "SELECT strategy_id, COUNT(*) AS n, "
            "SUM(CAST(strategy_edge AS REAL)) AS sum_edge, "
            "SUM(CAST(execution_cost AS REAL)) AS sum_exec, "
            "SUM(CAST(short_term_alpha AS REAL)) AS sum_alpha, "
            "SUM(CAST(fee AS REAL)) AS sum_fee, "
            "SUM(CAST(cost_basis_dollars AS REAL)) AS sum_cost_basis, "
            "AVG(CAST(venue_fee_bps AS REAL)) AS avg_fee_bps "
            "FROM attribution WHERE fill_ts_ns >= ?"
        )
        params: list[Any] = [int(since_ns)]
        if strategy_id is not None:
            sql += " AND strategy_id = ?"
            params.append(strategy_id)
        sql += " GROUP BY strategy_id"
        rows = self._conn.execute(sql, params).fetchall()
        out = {}
        total_n = 0
        for r in rows:
            sid, n, e, c, a, fee, cost_basis, avg_bps = r
            out[sid] = {
                "fills": int(n),
                "strategy_edge_sum": float(e or 0.0),
                "execution_cost_sum": float(c or 0.0),
                "short_term_alpha_sum": float(a or 0.0),
                "fee_sum": float(fee or 0.0),
                "total_cost_basis_dollars": float(cost_basis or 0.0),
                "avg_venue_fee_bps": float(avg_bps or 0.0),
            }
            total_n += int(n)
        return {"strategies": out, "total_fills": total_n, "since_ns": int(since_ns)}

    def get_record(self, fill_id: str) -> AttributionRecord | None:
        row = self._conn.execute(
            "SELECT fill_id, intent_id, leg_id, strategy_id, venue, market_id, "
            "side, size, intent_price, decision_price, arrival_price, fill_price, "
            "exit_price, strategy_edge, execution_cost, short_term_alpha, fee, "
            "cost_basis_dollars, venue_fee_bps, "
            "fill_ts_ns, settled_ts_ns, extra_json FROM attribution WHERE fill_id=?",
            (fill_id,),
        ).fetchone()
        if row is None:
            return None
        return AttributionRecord(
            fill_id=row[0],
            intent_id=row[1],
            leg_id=row[2],
            strategy_id=row[3],
            venue=row[4],
            market_id=row[5],
            side=Side(row[6]),
            size=Decimal(row[7]),
            intent_price=Decimal(row[8]) if row[8] is not None else None,
            decision_price=Decimal(row[9]) if row[9] is not None else None,
            arrival_price=Decimal(row[10]) if row[10] is not None else None,
            fill_price=Decimal(row[11]),
            exit_price=Decimal(row[12]) if row[12] is not None else None,
            strategy_edge=Decimal(row[13]) if row[13] is not None else None,
            execution_cost=Decimal(row[14]) if row[14] is not None else None,
            short_term_alpha=Decimal(row[15]) if row[15] is not None else None,
            fee=Decimal(row[16]) if row[16] is not None else None,
            cost_basis_dollars=Decimal(row[17]) if row[17] is not None else None,
            venue_fee_bps=Decimal(row[18]) if row[18] is not None else None,
            fill_ts_ns=int(row[19]),
            settled_ts_ns=int(row[20]),
            extra=json.loads(row[21] or "{}"),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _upsert(self, rec: AttributionRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO attribution "
            "(fill_id, intent_id, leg_id, strategy_id, venue, market_id, side, "
            " size, intent_price, decision_price, arrival_price, fill_price, "
            " exit_price, strategy_edge, execution_cost, short_term_alpha, fee, "
            " cost_basis_dollars, venue_fee_bps, "
            " fill_ts_ns, settled_ts_ns, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec.fill_id,
                rec.intent_id,
                rec.leg_id,
                rec.strategy_id,
                rec.venue,
                rec.market_id,
                rec.side.value,
                str(rec.size),
                str(rec.intent_price) if rec.intent_price is not None else None,
                str(rec.decision_price) if rec.decision_price is not None else None,
                str(rec.arrival_price) if rec.arrival_price is not None else None,
                str(rec.fill_price),
                str(rec.exit_price) if rec.exit_price is not None else None,
                str(rec.strategy_edge) if rec.strategy_edge is not None else None,
                str(rec.execution_cost) if rec.execution_cost is not None else None,
                str(rec.short_term_alpha) if rec.short_term_alpha is not None else None,
                str(rec.fee) if rec.fee is not None else None,
                str(rec.cost_basis_dollars) if rec.cost_basis_dollars is not None else None,
                str(rec.venue_fee_bps) if rec.venue_fee_bps is not None else None,
                int(rec.fill_ts_ns),
                int(rec.settled_ts_ns),
                json.dumps(rec.extra, default=str),
            ),
        )
        self._conn.commit()


def _signed(side: Side, raw: Decimal) -> Decimal:
    """
    Sign convention: positive = adverse for the strategy.
    For BUY: paying more / mid going down post-fill is adverse.
    For SELL: paying less / mid going up post-fill is adverse.

    `raw` is computed in the BUY-frame (high-to-low movement = positive).
    For SELL we negate so that positive remains "adverse to the strategy".
    """
    return raw if side == Side.BUY else -raw
