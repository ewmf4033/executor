# Polymarket recorder — scaffold (0a2+ reference)

**Status:** REFERENCE-ONLY. Not wired, no service, not invoked by the executor.
This directory exists so that when venue expansion to Polymarket is prioritized
(post-Build-Zero), the team isn't starting from a blank page.

Do NOT `systemctl enable` anything in this directory. Do NOT subscribe the
executor to a Polymarket feed from here. If that time comes, the scaffold should
be promoted into `data-recorder` (or its own service) with its own ops plan.

## Relationship to existing recorders

The 0a1 recorder at `/root/data-recorder/` owns Kalshi capture. Per repo hard
constraints, that directory is not modified. When Polymarket onboarding begins,
decide first whether data-recorder's module layout supports a second venue
cleanly, or whether a sibling `polymarket-recorder` service is cleaner.

## Polymarket WS delta vs Kalshi

| Concern            | Kalshi                          | Polymarket (CLOB)               |
|--------------------|---------------------------------|---------------------------------|
| WS endpoint        | `wss://api.elections.kalshi.com/trade-api/ws/v2` | `wss://ws-subscriptions-clob.polymarket.com/ws/market` |
| Auth               | Ed25519 signed header per-req   | None for public market feeds; API key required for user channel |
| Subscribe          | `{type: "subscribe", params: {channels: [...], market_tickers: [...]}}` | `{assets_ids: [...], type: "market"}` |
| Orderbook msg      | `orderbook_delta` + `orderbook_snapshot` | `book` (full) + `price_change` + `tick_size_change` |
| Trades             | `trade` event                   | `trade` (structure differs: `maker_orders` array) |
| Identifier         | `ticker` (e.g. `PRESWIN-24NOV-DT`) | `asset_id` (tokenId uint256 hex) + `condition_id` |
| Outcome shape      | YES / NO markets                | multi-outcome ERC-1155 tokens mapped to YES/NO per condition |
| Heartbeat          | Server pings every 10s          | Client expected to `ping` every ~30s |
| Price ticks        | 1c or 0.5c                      | 1c, configurable per-market `tick_size` |

Key operational differences:

1. **Identifier mapping.** Polymarket markets are identified by on-chain
   `condition_id` and resolved via CLOB REST (`/markets`). We need a
   one-time (or daily) pull to map tokenIds to human labels. Kalshi ticker
   is self-describing.
2. **No signed WS for public market data.** Simplifies auth plumbing.
3. **Tick-size change events.** Polymarket can change the price tick mid-life;
   the recorder must handle the `tick_size_change` message and re-bucket
   orderbook snapshots, or storage will misalign.
4. **Reconnect & snapshot.** Polymarket does not replay missed messages on
   reconnect; a fresh `book` is guaranteed first on re-subscribe. Gap
   detection must treat the reconnect snapshot as authoritative.

## What this scaffold provides

- `poly_ws_stub.py` — connection/reconnect/parquet-hook skeleton matching
  the 0a1 pattern. Does NOT connect to live Polymarket. Methods raise
  `NotImplementedError` where a real implementation is needed.
- This README — the delta-from-Kalshi notes above, so future-us doesn't
  rediscover them by reading Polymarket docs under time pressure.

## Open questions for Phase 5+

1. **Storage co-location.** Does Polymarket parquet live under the same
   B2 bucket + prefix layout as Kalshi, or a sibling bucket?
2. **Timestamps.** Polymarket WS uses `timestamp` in seconds; 0a1 uses ns.
   Normalize at the recorder boundary or downstream?
3. **Canonicalization.** Kalshi `ticker` and Polymarket `condition_id`
   need a canonical `market_key` in the executor's intent schema if a
   strategy spans both venues (e.g. arbitrage).
