# Manual Execution Journal

Purpose:
Track real/manual trades taken from alerts so Tier 1 readiness can be evaluated from actual executed entries, not just generated signals.

This is manual-only.

Safety boundaries:
- Does not place orders.
- Does not call Kalshi APIs.
- Does not read API keys.
- Does not touch daemon/systemd/env/auth/adapter code.
- Does not integrate with executor.service.
- Does not automate trading.

Required fields:
- timestamp_utc
- system
- market_ticker
- side
- entry_price
- size_dollars
- alert_edge
- reason
- notes

Current limitation:
This is entry logging only. It does not yet track closing price, settlement, CLV, P&L, or reconciliation.
