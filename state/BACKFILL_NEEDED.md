# Backfill: cost_basis_dollars and venue_fee_bps

Rows inserted before the 2026-04-21 migration have NULL for
`cost_basis_dollars` and `venue_fee_bps`.

To backfill BUY rows:

```sql
UPDATE attribution
SET cost_basis_dollars = CAST(size AS REAL) * CAST(fill_price AS REAL)
                         + COALESCE(CAST(fee AS REAL), 0)
WHERE cost_basis_dollars IS NULL AND side = 'BUY';
```

To backfill SELL rows:

```sql
UPDATE attribution
SET cost_basis_dollars = CAST(size AS REAL) * CAST(fill_price AS REAL)
                         - COALESCE(CAST(fee AS REAL), 0)
WHERE cost_basis_dollars IS NULL AND side = 'SELL';
```

To backfill venue_fee_bps (all sides):

```sql
UPDATE attribution
SET venue_fee_bps = (CAST(fee AS REAL) / (CAST(size AS REAL) * CAST(fill_price AS REAL))) * 10000
WHERE venue_fee_bps IS NULL
  AND fee IS NOT NULL
  AND CAST(size AS REAL) * CAST(fill_price AS REAL) > 0;
```
