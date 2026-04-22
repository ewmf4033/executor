# Backfill: cost_basis_dollars and venue_fee_bps — COMPLETED 2026-04-22

Completed in Phase 4.10 via `scripts/backfill_attribution.py`.

All 328 pre-migration rows had NULL `cost_basis_dollars` and `venue_fee_bps`
populated from `size`, `fill_price`, `fee`, and `side`. Post-backfill:

```
sqlite3 state/attribution.sqlite \
  "SELECT COUNT(*) FROM attribution WHERE cost_basis_dollars IS NULL"
# -> 0
```

The script is idempotent (COALESCE, only touches NULL rows) and remains in
the repo so the same procedure applies if another migration introduces new
nullable-at-add columns in the future.

Log: /var/log/attribution-backfill.log
