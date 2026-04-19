# executor

Build Zero item **0d**: the executor module. Load-bearing wall for every Tier 1+ strategy.

Spec: `/root/trading-wiki/specs/0d-executor.md` (version 2026-04-19).
Decisions deviating from that spec (only if forced by implementation reality): see `DECISIONS.md`.

## Layout

```
executor/
├── core/             # event bus, intent model, orchestration loop, data types
├── audit/            # append-only SQLite event log, daily rotation
├── strategies/       # Strategy abstract base + implementations
├── venue_adapters/   # VenueAdapter abstract base + per-venue implementations
├── risk/             # 13 gates + config + state (Phase 3)
└── kill/             # kill switch state + Telegram bot (Phase 4)
```

## Status

- **Phase 1 (current):** Scaffolding, event bus, audit log, data classes, abstract bases. No venue adapter, no risk gates.
- Phase 2: Kalshi paper adapter.
- Phase 3: Risk policy + 0g data poisoning detector.
- Phase 4: Kill switch + Telegram + 0e/0f detectors + example strategy.

## Run

```bash
cd /root/executor
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m executor               # run service
.venv/bin/python -m pytest                  # run tests
.venv/bin/python scripts/smoke_phase1.py    # phase-1 smoke: fake intent -> audit log
```

## Systemd

Install with:
```bash
sudo cp systemd/executor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now executor
journalctl -u executor -f
```

Not enabled by default during Phase 1.
