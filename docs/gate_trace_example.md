# Gate Trace Example — one concrete intent through all 13 risk gates

This document walks a single synthetic intent through the full risk-policy pipeline
and shows exactly what every gate sees, decides, and costs in wall-clock time.
It exists so a human (or a future me) can verify that the sequential gate order
described in `/root/trading-wiki/specs/0d-executor.md` (Decision 3) matches what
the code actually does.

**Reproducing.** The transcript below is emitted verbatim by

```
.venv/bin/python scripts/trace_one_intent.py
```

Re-run that script after any risk-module change; if the trace here no longer
matches, update this file.

## The intent

```
strategy_id = "YESNOCrossDetect"
leg         = BUY 800 YES @ 0.56 on kalshi:KALSHI-PRES-2028-DEM
              confidence=0.72  edge=0.04  horizon=600s  caps=(supports_limit,)
created_ts  = now                expires_ts = now + 300s
notional    = 800 * 0.56 = $448.00
```

## Pre-evaluation state

| Key                                                           | Value                             |
|---------------------------------------------------------------|-----------------------------------|
| market ceiling (default)                                      | $500.00                           |
| per-intent cap (default)                                      | $250.00                           |
| clip floor (default)                                          | 0.5 (final/original notional)     |
| current exposure on `kalshi:KALSHI-PRES-2028-DEM:YES`         | **$300.00** (seed)                |
| event_id map `(kalshi,KALSHI-PRES-2028-DEM) -> PRES-2028`     | registered                        |
| orderbook (YES asks top-3)                                    | 0.56/80, 0.57/60, 0.58/50 → depth 190 |
| kill state, venue_health, poisoning, adverse_selection        | clean                             |
| daily_pnl(YESNOCrossDetect)                                   | 0                                 |

## Trace

Columns: **`gate — order`**, input size (contracts entering the gate),
decision, resulting reason, wall-clock cost.

| # | Gate (order)           | Input size | Decision | Reason                                                                 | Cost     |
|---|------------------------|-----------:|----------|-------------------------------------------------------------------------|---------:|
| 1 | `structural` (1)       | 800        | APPROVE  | schema valid, market in universe, caps present, edge 0.04 ≥ 0.01 default | 0.026 ms |
| 2 | `kill_switch` (2)      | 800        | APPROVE  | no GLOBAL / STRATEGY / VENUE / STRATEGY_VENUE entries engaged             | 0.016 ms |
| 3 | `venue_health` (2.5)   | 800        | APPROVE  | kalshi incidents=0, not paused                                            | 0.010 ms |
| 4 | `poisoning` (2.6)      | 800        | APPROVE  | 0g tracker reports no pause on KALSHI-PRES-2028-DEM                       | 0.009 ms |
| 5 | `adverse_selection` (3)| 800        | APPROVE  | `NullAdverseSelectionDetector` (0e Phase 4) returns False                 | 0.008 ms |
| 6 | `per_intent_dollar_cap` (4) | 800   | **CLIP** | intent notional $448.00 > cap $250.00 → ratio 250/448 = 0.558 → 446 ctr  | 0.118 ms |
| 7 | `liquidity` (4.5)      | 446        | **CLIP** | top-3 YES ask depth = 80+60+50 = 190 contracts; 446 > 190 → clip to 190   | 0.059 ms |
| 8 | `market_exposure` (5)  | 190        | APPROVE  | current $300 + 190·0.56=$106.40 = $406.40 < ceiling $500                  | 0.014 ms |
| 9 | `event_concentration` (5.5) | 190   | APPROVE  | event `PRES-2028` total exposure after add = $406.40 < $1,000 default    | 0.018 ms |
|10 | `venue_exposure` (6)   | 190        | APPROVE  | kalshi venue total $406.40 < $2,500 default                              | 0.018 ms |
|11 | `global_portfolio` (7) | 190        | APPROVE  | global total $406.40 < $10,000 default                                    | 0.012 ms |
|12 | `strategy_allocation` (7.5) | 190  | APPROVE  | strategy `YESNOCrossDetect` allocation $106.40 < $1,000 default          | 0.011 ms |
|13 | `daily_loss` (8)       | 190        | APPROVE  | daily_pnl(YESNOCrossDetect) = $0 > -$200 default                         | 0.174 ms |
|14 | `clip_floor` (9)       | 190        | **REJECT** | RISK_CLIPPED: final/original = 190/800 = 0.238 < 0.5 floor               | 0.019 ms |

## Audit event stream

Two `GATE_CLIPPED` events precede the terminal `GATE_REJECTED`:

```
GATE_CLIPPED  gate=per_intent_dollar_cap   800 → 446   reason="intent notional 448.00 > cap 250.00 …"
GATE_CLIPPED  gate=liquidity               446 → 190   reason="liquidity: clipped to top-N depth"
GATE_REJECTED gate=clip_floor              reason="RISK_CLIPPED: final/original 0.238 < 0.5"
              gates_passed=[structural, kill_switch, venue_health, poisoning,
                            adverse_selection, per_intent_dollar_cap, liquidity,
                            market_exposure, event_concentration, venue_exposure,
                            global_portfolio, strategy_allocation, daily_loss]
```

## What this confirms about the design

1. **Sequential execution.** Each gate consumes the output of the previous
   (note how the `liquidity` clip pushes input into `market_exposure` from 446
   down to 190, and the rest of the pipeline sees 190).
2. **CLIPs accumulate; REJECTs short-circuit.** `per_intent_dollar_cap` and
   `liquidity` both clipped without stopping the pipeline. Only when the
   compound clip ratio crossed the `clip_floor` did the pipeline emit REJECT.
3. **Cumulative clip guard is last, not first.** `clip_floor` runs at gate 9
   (position 14 by index) so it sees the final compound size, not any
   intermediate state.
4. **Gate timings are recorded individually.** Each gate's wall-clock cost is
   captured in `gate_timings_ms`. The sum for this intent is ~0.5 ms, dominated
   by `daily_loss` and `per_intent_dollar_cap`. Every entry in the table above
   is emitted verbatim into the `INTENT_ADMITTED` or `GATE_REJECTED` payload.
5. **`gates_passed` is an ordered trace, not a set.** The REJECT payload lists
   13 passed gates (all of 1…8) in the exact order they approved/clipped, so a
   downstream consumer can reconstruct the pipeline state.

## When to re-run this

Any change to:

- `executor/risk/gates.py` — a gate's math or ordering
- `executor/risk/policy.py` — the orchestration loop
- `executor/risk/exposure.py` — the notional formula
- default values in `executor/risk/config.py`

…requires regenerating this trace. The script is intentionally hermetic — no
venue, no network — so the output is reproducible across machines.
