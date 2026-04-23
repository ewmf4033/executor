#!/usr/bin/env python3
"""
Daily Telegram digest for the executor trading system.

Collects pipeline stats, fill attribution, safety signals, and system
health from the prior 24h (UTC midnight to midnight) and sends a
formatted summary via Telegram.

Standalone script — no executor imports required.

Exit codes: 0 = success, 1 = error.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import errno
import fcntl
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


DEFAULT_LOCKFILE = "/var/run/executor-digest.lock"
DEFAULT_TG_FAIL_FLAG = "/var/run/executor-digest-telegram-last-failed"


@contextlib.contextmanager
def _acquire_lock(lockfile: str):
    """Acquire an exclusive non-blocking flock on `lockfile`.

    Yields True if the lock was acquired, False if another digest is
    already running. Does not raise on EACCES (e.g., unwritable
    /var/run in tests) — logs a warning and proceeds without locking.
    """
    fd = None
    got_lock = False
    try:
        try:
            fd = os.open(lockfile, os.O_CREAT | os.O_WRONLY, 0o644)
        except OSError as exc:
            print(
                f"warn: cannot open lockfile {lockfile}: {exc}; proceeding without lock",
                file=sys.stderr,
            )
            yield True
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            got_lock = True
            yield True
        except OSError as exc:
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                yield False
            else:
                raise
    finally:
        if fd is not None:
            if got_lock:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except Exception:
                    pass
            try:
                os.close(fd)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ns_range_for_date(date: dt.date) -> tuple[int, int]:
    """Return (start_ns_inclusive, end_ns_exclusive) for a UTC calendar day."""
    start = int(dt.datetime(date.year, date.month, date.day, tzinfo=dt.timezone.utc).timestamp() * 1e9)
    end = int(dt.datetime(date.year, date.month, date.day, tzinfo=dt.timezone.utc).timestamp() * 1e9) + 86_400_000_000_000
    return start, end


def _open_db(path: str) -> sqlite3.Connection | None:
    """Open a SQLite DB read-only; return None if missing."""
    if not Path(path).exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _send_telegram(text: str, token: str, chat_id: str) -> None:
    """Send a Telegram message via urllib (MarkdownV2 with fallback)."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Try Markdown first, fall back to plain text on parse errors.
    for parse_mode in ("Markdown", None):
        payload = json.dumps({
            "chat_id": chat_id,
            "text": text,
            **({"parse_mode": parse_mode} if parse_mode else {}),
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read())
                if body.get("ok"):
                    return
        except Exception:
            if parse_mode is None:
                raise
            continue  # retry without parse_mode


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_audit_data(audit_dir: str, target_date: dt.date) -> dict:
    """Collect event stats from the audit DB for target_date."""
    db_name = f"audit-{target_date.isoformat()}.sqlite"
    db_path = Path(audit_dir) / db_name
    start_ns, end_ns = _ns_range_for_date(target_date)

    result: dict = {
        "event_counts": {},
        "top_rejections": [],
        "kill_triggers": 0,
        "self_check_ok": 0,
        "self_check_fail": 0,
        "error_count": 0,
    }

    conn = _open_db(str(db_path))
    if conn is None:
        return result

    try:
        # 1. Event counts by type
        for row in conn.execute(
            "SELECT event_type, COUNT(*) as cnt FROM events WHERE ts_ns >= ? AND ts_ns < ? GROUP BY event_type",
            (start_ns, end_ns),
        ):
            result["event_counts"][row["event_type"]] = row["cnt"]

        # 2. Gate rejection breakdown
        rows = conn.execute(
            "SELECT json_extract(payload_json, '$.gate') as gate, COUNT(*) as cnt "
            "FROM events WHERE event_type='GATE_REJECTED' AND ts_ns >= ? AND ts_ns < ? "
            "GROUP BY gate ORDER BY cnt DESC LIMIT 5",
            (start_ns, end_ns),
        ).fetchall()
        result["top_rejections"] = [(r["gate"], r["cnt"]) for r in rows]

        # 3-5. Specific counts
        result["kill_triggers"] = result["event_counts"].get("KILL_STATE_CHANGED", 0)
        result["self_check_ok"] = result["event_counts"].get("SELF_CHECK_OK", 0)
        result["self_check_fail"] = result["event_counts"].get("SELF_CHECK_FAIL", 0)
        result["error_count"] = result["event_counts"].get("ERROR", 0)
    finally:
        conn.close()

    return result


def collect_attribution_data(attr_db: str, target_date: dt.date) -> dict:
    """Collect fill stats from the attribution DB for target_date."""
    start_ns, end_ns = _ns_range_for_date(target_date)

    result: dict = {
        "fill_count": 0,
        "fills_by_venue": {},
        "paper_pnl": 0.0,
        "total_cost_basis": 0.0,
        "total_fees": 0.0,
    }

    conn = _open_db(attr_db)
    if conn is None:
        return result

    try:
        # Fill count per venue
        for row in conn.execute(
            "SELECT venue, COUNT(*) as cnt FROM attribution "
            "WHERE fill_ts_ns >= ? AND fill_ts_ns < ? GROUP BY venue",
            (start_ns, end_ns),
        ):
            result["fills_by_venue"][row["venue"]] = row["cnt"]
        result["fill_count"] = sum(result["fills_by_venue"].values())

        # Gross paper P&L — direction-aware true realized PnL.
        # Phase 4.11.3: mirrors the fix in AttributionTracker.settle_due
        # (Phase 4.11.2). short_term_alpha uses adverse-positive convention
        # and is NOT a PnL feed; true realized PnL is:
        #   BUY:  (exit_price - fill_price) * size
        #   SELL: (fill_price - exit_price) * size
        # Rows with NULL exit_price (unsettled fills) are excluded to mirror
        # the fail-closed guard in tracker.py:345-349.
        row = conn.execute(
            "SELECT COALESCE(SUM("
            "  CASE "
            "    WHEN side = 'BUY' THEN (CAST(exit_price AS REAL) - CAST(fill_price AS REAL)) * CAST(size AS REAL) "
            "    WHEN side = 'SELL' THEN (CAST(fill_price AS REAL) - CAST(exit_price AS REAL)) * CAST(size AS REAL) "
            "    ELSE 0 "
            "  END"
            "), 0.0) as pnl "
            "FROM attribution "
            "WHERE fill_ts_ns >= ? AND fill_ts_ns < ? "
            "AND exit_price IS NOT NULL AND fill_price IS NOT NULL",
            (start_ns, end_ns),
        ).fetchone()
        result["paper_pnl"] = round(float(row["pnl"]), 4) if row else 0.0

        # Total cost basis
        row = conn.execute(
            "SELECT COALESCE(SUM(ABS(CAST(fill_price AS REAL) * CAST(size AS REAL))), 0.0) as notional, "
            "       COALESCE(SUM(CAST(COALESCE(fee, '0') AS REAL)), 0.0) as fees "
            "FROM attribution WHERE fill_ts_ns >= ? AND fill_ts_ns < ?",
            (start_ns, end_ns),
        ).fetchone()
        result["total_cost_basis"] = round(float(row["notional"]), 2) if row else 0.0
        result["total_fees"] = round(float(row["fees"]), 4) if row else 0.0
    finally:
        conn.close()

    return result


def collect_system_data() -> dict:
    """Best-effort system health checks."""
    result: dict = {"executor_uptime_h": None, "data_recorder_status": None}

    # Executor uptime
    try:
        out = subprocess.run(
            ["systemctl", "show", "executor", "--property=ActiveEnterTimestamp"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and "=" in out.stdout:
            ts_str = out.stdout.strip().split("=", 1)[1].strip()
            if ts_str:
                # Parse systemd timestamp like "Mon 2026-04-20 08:00:00 UTC"
                for fmt in ("%a %Y-%m-%d %H:%M:%S %Z", "%Y-%m-%d %H:%M:%S %Z"):
                    try:
                        entered = dt.datetime.strptime(ts_str, fmt).replace(tzinfo=dt.timezone.utc)
                        delta = dt.datetime.now(dt.timezone.utc) - entered
                        result["executor_uptime_h"] = round(delta.total_seconds() / 3600, 1)
                        break
                    except ValueError:
                        continue
    except Exception:
        pass

    # data-recorder status
    try:
        out = subprocess.run(
            ["systemctl", "is-active", "data-recorder"],
            capture_output=True, text=True, timeout=5,
        )
        result["data_recorder_status"] = out.stdout.strip() if out.returncode == 0 else "inactive"
    except Exception:
        pass

    return result


def load_rolling_history(digest_dir: str, target_date: dt.date, today_data: dict) -> list[dict]:
    """Load prior 6 daily digest JSONs + today to build 7-day trend."""
    trend = []
    for i in range(6, 0, -1):
        d = target_date - dt.timedelta(days=i)
        path = Path(digest_dir) / f"digest-{d.isoformat()}.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                trend.append({
                    "date": d.isoformat(),
                    "day": d.strftime("%a"),
                    "emitted": data.get("event_counts", {}).get("INTENT_EMITTED", 0),
                    "admitted": data.get("event_counts", {}).get("INTENT_ADMITTED", 0),
                    "rejected": data.get("event_counts", {}).get("GATE_REJECTED", 0),
                    "fills": data.get("fill_count", 0),
                })
            except Exception:
                continue
    # Today's entry
    trend.append({
        "date": target_date.isoformat(),
        "day": target_date.strftime("%a"),
        "emitted": today_data.get("event_counts", {}).get("INTENT_EMITTED", 0),
        "admitted": today_data.get("event_counts", {}).get("INTENT_ADMITTED", 0),
        "rejected": today_data.get("event_counts", {}).get("GATE_REJECTED", 0),
        "fills": today_data.get("fill_count", 0),
    })
    return trend


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_message(target_date: dt.date, audit: dict, attr: dict,
                   system: dict, trend: list[dict]) -> str:
    """Build the Telegram digest message (Markdown, <4000 chars)."""
    ec = audit["event_counts"]
    emitted = ec.get("INTENT_EMITTED", 0)
    admitted = ec.get("INTENT_ADMITTED", 0)
    rejected = ec.get("GATE_REJECTED", 0)
    crashed = audit["error_count"]
    admit_rate = (admitted / emitted * 100) if emitted > 0 else 0.0

    lines = [
        f"*Daily Digest — {target_date.isoformat()}*",
        "",
        "📊 *Pipeline*",
        f"Emitted: {emitted} | Admitted: {admitted} | Rejected: {rejected} | Crashed: {crashed}",
        f"Admit rate: {admit_rate:.1f}%",
        "",
    ]

    # Top rejections
    lines.append("🚫 *Top Rejections*")
    if audit["top_rejections"]:
        for i, (gate, cnt) in enumerate(audit["top_rejections"][:5], 1):
            lines.append(f"{i}. {gate}: {cnt}")
    else:
        lines.append("None")
    lines.append("")

    # Fills
    lines.append("💰 *Fills*")
    notional_str = f"${attr['total_cost_basis']:,.2f}"
    fees_str = f"${attr['total_fees']:.2f}"
    pnl_str = f"${attr['paper_pnl']:.2f}"
    lines.append(f"Count: {attr['fill_count']} | Notional: {notional_str} | Fees: {fees_str}")
    lines.append(f"Paper P&L (α): {pnl_str}")
    if attr["fills_by_venue"]:
        venue_parts = [f"{v}: {c}" for v, c in sorted(attr["fills_by_venue"].items())]
        lines.append(f"Venues: {', '.join(venue_parts)}")
    lines.append("")

    # Safety
    lines.append("🔒 *Safety*")
    sc_status = "OK ✓" if audit["self_check_fail"] == 0 and audit["self_check_ok"] > 0 else (
        "FAIL ✗" if audit["self_check_fail"] > 0 else "N/A"
    )
    lines.append(f"Self-check: {sc_status} | Kill triggers: {audit['kill_triggers']}")
    lines.append(f"Errors: {audit['error_count']}")
    lines.append("")

    # System
    lines.append("⏱ *System*")
    uptime_str = f"up {system['executor_uptime_h']}h" if system.get("executor_uptime_h") is not None else "unknown"
    recorder_str = system.get("data_recorder_status") or "unknown"
    lines.append(f"Executor: {uptime_str} | data-recorder: {recorder_str}")
    lines.append("")

    # 7-day trend
    if trend:
        lines.append("📈 *7-day trend*")
        lines.append("```")
        lines.append(f"{'Day':<4} | {'Emit':>5} | {'Admit':>5} | {'Rej':>5} | {'Fills':>5}")
        for t in trend:
            lines.append(f"{t['day']:<4} | {t['emitted']:>5} | {t['admitted']:>5} | {t['rejected']:>5} | {t['fills']:>5}")
        lines.append("```")

    msg = "\n".join(lines)
    # Truncate if somehow over 4000 chars
    if len(msg) > 3950:
        msg = msg[:3950] + "\n..."
    return msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Daily executor digest")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print, don't send Telegram")
    parser.add_argument("--audit-dir", default="/root/executor/audit-logs/paper_live",
                        help="Directory containing audit-YYYY-MM-DD.sqlite files")
    parser.add_argument("--attr-db", default="/root/executor/state/attribution.sqlite",
                        help="Path to attribution.sqlite")
    parser.add_argument("--digest-dir", default="/root/executor/audit-logs/digests",
                        help="Directory to save digest JSON files")
    parser.add_argument("--date", default=None,
                        help="Target date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--lockfile", default=DEFAULT_LOCKFILE,
                        help="Lockfile path for run-at-a-time enforcement")
    parser.add_argument("--tg-fail-flag", default=DEFAULT_TG_FAIL_FLAG,
                        help="Flag file written when Telegram send fails")
    args = parser.parse_args()

    # Determine target date (yesterday by default)
    if args.date:
        target_date = dt.date.fromisoformat(args.date)
    else:
        target_date = dt.date.today() - dt.timedelta(days=1)

    # Phase 4.9 Item 4: serialize runs via flock. A concurrent run exits
    # cleanly (0) — not an error — since the other process is doing the work.
    with _acquire_lock(args.lockfile) as got_lock:
        if not got_lock:
            print("digest already running, exiting", file=sys.stderr)
            return 0

        # Phase 4.9 Item 5: data-collection / write failures return 1;
        # Telegram notification failures do NOT — they log a flag file.
        try:
            audit = collect_audit_data(args.audit_dir, target_date)
            attr = collect_attribution_data(args.attr_db, target_date)
            system = collect_system_data() if not args.dry_run else {
                "executor_uptime_h": None, "data_recorder_status": None,
            }
            trend = load_rolling_history(args.digest_dir, target_date,
                                         {**audit, "fill_count": attr["fill_count"]})
            message = format_message(target_date, audit, attr, system, trend)

            # Save digest JSON — atomic temp-rename so a concurrent reader
            # never sees a half-written file.
            digest_data = {
                **audit,
                **{k: v for k, v in attr.items()},
                "system": system,
                "target_date": target_date.isoformat(),
            }
            digest_path = Path(args.digest_dir)
            digest_path.mkdir(parents=True, exist_ok=True)
            json_file = digest_path / f"digest-{target_date.isoformat()}.json"
            _write_json_atomic(json_file, digest_data)
        except Exception as exc:
            print(f"ERROR: digest collection/write failed: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 1

        # Output
        print(f"Digest for {target_date.isoformat()}")
        print("=" * 50)
        print(message)
        print("=" * 50)
        print(f"Digest JSON saved to {json_file}")

        # Telegram — best-effort; failure here does NOT fail the run.
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if token and chat_id and not args.dry_run:
            try:
                _send_telegram(message, token, chat_id)
                print("Telegram message sent.")
                # Clear the flag on success so monitors can notice the
                # recovery (persistent failure = flag remains).
                _clear_flag(args.tg_fail_flag)
            except Exception as exc:
                print(
                    f"warn: telegram send failed: {exc}; digest JSON was still written",
                    file=sys.stderr,
                )
                _write_flag(args.tg_fail_flag, str(exc))
        elif args.dry_run:
            print("Dry run — Telegram skipped.")
        else:
            print("Telegram env vars not set — skipping send.")

        return 0


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON via temp file + os.rename so readers never see a partial file."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        json.dump(data, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    os.rename(tmp.name, str(path))


def _write_flag(flag_path: str, reason: str) -> None:
    try:
        p = Path(flag_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {reason}\n")
    except Exception as exc:
        print(f"warn: could not write flag {flag_path}: {exc}", file=sys.stderr)


def _clear_flag(flag_path: str) -> None:
    try:
        Path(flag_path).unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
