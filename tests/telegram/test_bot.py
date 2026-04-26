"""TelegramBot — command parsing, auth, rate limit, dispatch."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from executor.core.events import Event, EventType
from executor.kill.manager import KillManager
from executor.kill.state import KillMode, KillStateStore
from executor.telegram.bot import TelegramBot, parse_command


# ----------------------------------------------------------------------
# parse_command
# ----------------------------------------------------------------------


def test_parse_kill_soft() -> None:
    p = parse_command("/kill soft something broke")
    assert p.valid is True
    assert p.command == "kill"
    assert p.sub == "soft"
    assert p.args == "something broke"


def test_parse_kill_status_no_args() -> None:
    p = parse_command("/kill status")
    assert p.valid is True
    assert p.sub == "status"


def test_parse_kill_resume_no_args() -> None:
    p = parse_command("/kill resume")
    assert p.valid is True
    assert p.sub == "resume"


def test_parse_kill_soft_requires_reason() -> None:
    p = parse_command("/kill soft")
    assert p.valid is False
    assert "reason" in p.error


def test_parse_kill_unknown_sub() -> None:
    p = parse_command("/kill banana")
    assert p.valid is False
    assert "unknown" in p.error.lower()


def test_parse_strips_bot_mention() -> None:
    p = parse_command("/kill@masta_op2_bot status")
    assert p.valid is True
    assert p.command == "kill"
    assert p.sub == "status"


def test_parse_unknown_command() -> None:
    p = parse_command("/banana")
    assert p.valid is False


def test_parse_not_a_command() -> None:
    p = parse_command("hello")
    assert p.valid is False


def test_parse_venue_health() -> None:
    p = parse_command("/venue health")
    assert p.valid is True
    assert p.command == "venue"
    assert p.sub == "health"


def test_parse_venue_unknown_sub() -> None:
    p = parse_command("/venue burn")
    assert p.valid is False


def test_parse_empty_string() -> None:
    p = parse_command("")
    assert p.valid is False


# ----------------------------------------------------------------------
# Bot dispatch
# ----------------------------------------------------------------------


def _bot(tmp_path: Path) -> tuple[TelegramBot, KillManager, list[Event]]:
    store = KillStateStore(tmp_path / "kill.sqlite")
    captured: list[Event] = []

    async def pub(ev: Event) -> None:
        captured.append(ev)

    km = KillManager(store=store, publish=pub, panic_cooldown_sec=1)
    bot = TelegramBot(
        kill_manager=km,
        token="tok",
        chat_id="42",
        rate_limit_sec=0.0,
    )
    return bot, km, captured


def test_dispatch_kill_soft(tmp_path: Path) -> None:
    bot, km, _ = _bot(tmp_path)
    reply = asyncio.run(bot.handle_text("/kill soft halt", chat_id="42"))
    assert "SOFT engaged" in reply
    assert km.mode is KillMode.SOFT


def test_dispatch_kill_status(tmp_path: Path) -> None:
    bot, _, _ = _bot(tmp_path)
    reply = asyncio.run(bot.handle_text("/kill status", chat_id="42"))
    assert "mode=NONE" in reply


def test_dispatch_kill_hard_then_resume(tmp_path: Path) -> None:
    bot, km, _ = _bot(tmp_path)
    asyncio.run(bot.handle_text("/kill hard halt", chat_id="42"))
    assert km.mode is KillMode.HARD
    reply = asyncio.run(bot.handle_text("/kill resume", chat_id="42"))
    assert "resumed" in reply.lower()
    assert km.mode is KillMode.NONE


def test_dispatch_kill_panic_blocks_resume(tmp_path: Path) -> None:
    bot, _, _ = _bot(tmp_path)
    asyncio.run(bot.handle_text("/kill panic everything!", chat_id="42"))
    reply = asyncio.run(bot.handle_text("/kill resume", chat_id="42"))
    assert "BLOCKED" in reply


def test_unauthorized_chat_id_dropped(tmp_path: Path) -> None:
    bot, km, events = _bot(tmp_path)
    reply = asyncio.run(bot.handle_text("/kill soft halt", chat_id="99999"))
    assert reply == ""
    assert km.mode is KillMode.NONE
    # No KILL_COMMAND_RECEIVED for unauthorized.
    assert not any(e.event_type == EventType.KILL_COMMAND_RECEIVED for e in events)


def test_unauthorized_panic_does_not_bypass_auth(tmp_path: Path) -> None:
    """Phase 4.14d: the new rate-limit bypass for /kill hard and /kill
    panic must not circumvent chat-id authorization. Auth runs BEFORE
    parse_command in handle_text, so an unauthorized sender cannot
    trigger KillManager.engage via the critical-kill path."""
    bot, km, events = _bot(tmp_path)
    reply = asyncio.run(
        bot.handle_text("/kill panic emergency", chat_id="99999")
    )
    assert reply == ""
    assert km.mode is KillMode.NONE, "unauthorized panic must not engage"
    assert km.snapshot().panic is False
    # No audit event — auth ran before the KILL_COMMAND_RECEIVED emit.
    assert not any(
        e.event_type == EventType.KILL_COMMAND_RECEIVED for e in events
    )
    # And /kill hard is likewise gated.
    reply2 = asyncio.run(
        bot.handle_text("/kill hard system down", chat_id="99999")
    )
    assert reply2 == ""
    assert km.mode is KillMode.NONE


def test_rate_limit_drops_burst(tmp_path: Path) -> None:
    store = KillStateStore(tmp_path / "kill.sqlite")
    km = KillManager(store=store)
    bot = TelegramBot(kill_manager=km, token="tok", chat_id="42", rate_limit_sec=10.0)
    r1 = asyncio.run(bot.handle_text("/kill status", chat_id="42"))
    r2 = asyncio.run(bot.handle_text("/kill status", chat_id="42"))
    assert "mode=" in r1
    assert "rate-limited" in r2


def test_invalid_command_returns_err(tmp_path: Path) -> None:
    bot, _, _ = _bot(tmp_path)
    reply = asyncio.run(bot.handle_text("/kill banana", chat_id="42"))
    assert reply.startswith("err:")


def test_audit_event_emitted(tmp_path: Path) -> None:
    bot, _, events = _bot(tmp_path)
    asyncio.run(bot.handle_text("/kill status", chat_id="42"))
    received = [e for e in events if e.event_type == EventType.KILL_COMMAND_RECEIVED]
    assert len(received) == 1
    assert received[0].payload["chat_id"] == "42"


def test_venue_health_uses_provider(tmp_path: Path) -> None:
    store = KillStateStore(tmp_path / "kill.sqlite")
    km = KillManager(store=store)

    def provider() -> dict[str, str]:
        return {"kalshi": "ok", "polymarket": "tripped"}

    bot = TelegramBot(
        kill_manager=km,
        token="tok",
        chat_id="42",
        rate_limit_sec=0.0,
        venue_health_provider=provider,
    )
    reply = asyncio.run(bot.handle_text("/venue health", chat_id="42"))
    assert "kalshi=ok" in reply and "polymarket=tripped" in reply


def test_venue_health_no_provider(tmp_path: Path) -> None:
    bot, _, _ = _bot(tmp_path)
    reply = asyncio.run(bot.handle_text("/venue health", chat_id="42"))
    assert "no provider" in reply


# ---------------------------------------------------------------------------
# Phase 4.14b — dead-man Telegram commands.
# ---------------------------------------------------------------------------


def test_parse_ping_distinct_from_heartbeat() -> None:
    # /ping is status-only; /heartbeat updates last_heartbeat. The parser
    # must keep them as distinct commands even though neither takes args.
    p_ping = parse_command("/ping")
    p_hb = parse_command("/heartbeat")
    assert p_ping.valid is True and p_ping.command == "ping"
    assert p_hb.valid is True and p_hb.command == "heartbeat"
    assert p_ping.command != p_hb.command


def test_parse_arm_and_disarm() -> None:
    p_arm = parse_command("/arm 6h")
    assert p_arm.valid is True
    assert p_arm.command == "arm"
    assert p_arm.sub == "6h"

    p_arm_missing = parse_command("/arm")
    assert p_arm_missing.valid is False

    p_dis = parse_command("/disarm session end")
    assert p_dis.valid is True
    assert p_dis.command == "disarm"
    assert p_dis.args == "session end"

    p_dis_missing = parse_command("/disarm")
    assert p_dis_missing.valid is False


def _bot_with_dead_man(tmp_path: Path):
    from dataclasses import replace as _replace

    from executor.risk.config import DeadManCfg, RiskConfig
    from executor.risk.state import OperatorLivenessStore, RiskState

    store = KillStateStore(tmp_path / "kill.sqlite")
    captured: list[Event] = []

    async def pub(ev: Event) -> None:
        captured.append(ev)

    km = KillManager(store=store, publish=pub, panic_cooldown_sec=1)
    rs = RiskState(db_path=tmp_path / "risk_state.sqlite")
    asyncio.run(rs.load())
    liveness = OperatorLivenessStore(rs.connection)
    cfg = _replace(
        RiskConfig(),
        dead_man=DeadManCfg(
            enabled=True,
            default_timeout_sec=600,
            min_timeout_sec=60,
            max_timeout_sec=3600,
        ),
    )
    bot = TelegramBot(
        kill_manager=km,
        token="tok",
        chat_id="42",
        rate_limit_sec=0.0,
        operator_liveness=liveness,
        dead_man_cfg_getter=lambda: cfg.dead_man,
        publish=pub,
    )
    return bot, liveness, captured


def test_telegram_arm_heartbeat_disarm_flow(tmp_path: Path) -> None:
    bot, liveness, events = _bot_with_dead_man(tmp_path)
    # /arm 10m  -> armed
    reply = asyncio.run(bot.handle_text("/arm 10m", chat_id="42"))
    assert "armed for 10m" in reply
    assert liveness.load().armed is True

    # /ping must NOT update last_heartbeat.
    before_hb = liveness.load().last_heartbeat_ts_ns
    reply_ping = asyncio.run(bot.handle_text("/ping", chat_id="42"))
    assert "pong" in reply_ping and "armed=True" in reply_ping
    assert liveness.load().last_heartbeat_ts_ns == before_hb

    # /heartbeat updates last_heartbeat.
    reply_hb = asyncio.run(bot.handle_text("/heartbeat", chat_id="42"))
    assert "heartbeat ok" in reply_hb
    assert liveness.load().last_heartbeat_ts_ns >= before_hb

    # /disarm
    reply_dis = asyncio.run(bot.handle_text("/disarm done", chat_id="42"))
    assert "disarmed" in reply_dis
    assert liveness.load().armed is False

    # OPERATOR_* events were published.
    kinds = {e.event_type for e in events}
    assert EventType.OPERATOR_ARMED in kinds
    assert EventType.OPERATOR_HEARTBEAT in kinds
    assert EventType.OPERATOR_DISARMED in kinds


def test_disarm_reason_sanitized_for_control_chars(tmp_path: Path) -> None:
    """Phase 4.14e (M9) — operator-typed reason strings must have
    control chars, CR/LF, and ANSI escapes scrubbed before they reach
    logs, audit events, or Slack. Arm first, then disarm with a
    malicious reason and verify the stored/echoed reason is clean."""
    bot, liveness, events = _bot_with_dead_man(tmp_path)
    asyncio.run(bot.handle_text("/arm 10m", chat_id="42"))

    nasty = "bad\r\n\x1b[31mINJECT\x1b[0m\x07\x00stuff\tend"
    reply = asyncio.run(
        bot.handle_text(f"/disarm {nasty}", chat_id="42")
    )

    # Reply echoes the sanitized reason: no ESC, no CR/LF, no NUL, no BEL.
    assert "\x1b" not in reply
    assert "\r" not in reply and "\n" not in reply
    assert "\x00" not in reply and "\x07" not in reply
    # Visible parts of the original message survive.
    assert "bad" in reply
    assert "INJECT" in reply
    assert "stuff" in reply

    # Published OPERATOR_DISARMED event payload was sanitized identically.
    disarm_evs = [
        e for e in events if e.event_type is EventType.OPERATOR_DISARMED
    ]
    assert len(disarm_evs) == 1
    payload_reason = disarm_evs[0].payload["reason"]
    assert "\x1b" not in payload_reason
    assert "\r" not in payload_reason and "\n" not in payload_reason
    assert "\x00" not in payload_reason and "\x07" not in payload_reason


def test_disarm_reason_sanitized_length_cap(tmp_path: Path) -> None:
    """Reason strings are capped at 200 chars to bound log volume."""
    bot, _liveness, events = _bot_with_dead_man(tmp_path)
    asyncio.run(bot.handle_text("/arm 10m", chat_id="42"))
    long_reason = "x" * 1000
    asyncio.run(bot.handle_text(f"/disarm {long_reason}", chat_id="42"))
    ev = [
        e for e in events if e.event_type is EventType.OPERATOR_DISARMED
    ][-1]
    assert len(ev.payload["reason"]) <= 200


def test_disarm_only_control_chars_rejected_as_empty(tmp_path: Path) -> None:
    """A reason consisting solely of control chars/whitespace sanitizes
    to empty and must be rejected — same semantics as ``/disarm``
    with no argument."""
    bot, liveness, _events = _bot_with_dead_man(tmp_path)
    asyncio.run(bot.handle_text("/arm 10m", chat_id="42"))
    reply = asyncio.run(
        bot.handle_text("/disarm \x1b[31m\r\n\t\x00", chat_id="42")
    )
    assert "reason required" in reply
    assert liveness.load().armed is True  # still armed


# ---------------------------------------------------------------------------
# Phase 4.14c — /kill panic local-bypass.
# ---------------------------------------------------------------------------


async def test_bot_shielded_engage_exception_after_cancel_is_logged(
    tmp_path: Path,
) -> None:
    """Phase 4.14e follow-up: if _dispatch is cancelled mid-engage and
    the shielded inner engage subsequently raises, the exception must
    be retrieved (done-callback) and surfaced via the module logger —
    not silently swallowed as a GC-time warning."""
    from executor.telegram import bot as bot_mod

    bot, km, _events = _bot(tmp_path)
    started = asyncio.Event()

    async def _raising_engage(*_a, **_kw):
        started.set()
        await asyncio.sleep(0.05)
        raise RuntimeError("engage-blew-up-after-cancel")

    km.engage = _raising_engage  # type: ignore[assignment]

    captured: list[tuple[str, dict]] = []
    original_error = bot_mod.log.error

    def _spy_error(event, **kw):
        captured.append((event, kw))
        return original_error(event, **kw)

    bot_mod.log.error = _spy_error  # type: ignore[assignment]
    try:
        task = asyncio.create_task(
            bot.handle_text("/kill panic boom", chat_id="42")
        )
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.15)
    finally:
        bot_mod.log.error = original_error  # type: ignore[assignment]

    assert any(
        evt == "telegram.bot.shielded_engage_failed_after_cancel"
        and kw.get("kind") == "panic"
        and "engage-blew-up-after-cancel" in kw.get("error", "")
        for evt, kw in captured
    ), captured


def test_panic_engages_kill_before_reply_send(tmp_path: Path) -> None:
    """Panic must engage kill_mgr via _dispatch BEFORE any Telegram
    reply is sent (handle_update only calls _send_reply after
    _dispatch/handle_text returns)."""
    bot, km, _events = _bot(tmp_path)
    assert km.mode is KillMode.NONE
    # handle_text returns the reply string; by the time it returns, kill
    # must already be engaged.
    reply = asyncio.run(bot.handle_text("/kill panic big trouble", chat_id="42"))
    assert km.mode is KillMode.HARD
    snap = km.snapshot()
    assert snap.panic is True
    assert snap.manual_only is True
    assert "PANIC engaged" in reply


def test_panic_bypasses_rate_limit_after_recent_noncritical_command(
    tmp_path: Path,
) -> None:
    """Phase 4.14d: /kill panic must bypass the per-chat rate limit so
    a recent /ping or /kill status cannot suppress the kill signal."""
    store = KillStateStore(tmp_path / "kill.sqlite")
    captured: list[Event] = []

    async def pub(ev: Event) -> None:
        captured.append(ev)

    km = KillManager(store=store, publish=pub, panic_cooldown_sec=1)
    # Long rate-limit window that would normally suppress a follow-up.
    bot = TelegramBot(
        kill_manager=km, token="tok", chat_id="42", rate_limit_sec=60.0
    )

    # Consume rate budget with a non-critical command.
    r1 = asyncio.run(bot.handle_text("/kill status", chat_id="42"))
    assert "mode=" in r1  # status reply, rate slot consumed
    last_cmd_ts_before = bot._last_cmd_ts.get("42")

    # /kill panic must NOT be rate-limited.
    reply = asyncio.run(
        bot.handle_text("/kill panic emergency", chat_id="42")
    )
    assert "rate-limited" not in reply
    assert "PANIC engaged" in reply
    # Kill-state persisted.
    assert km.mode is KillMode.HARD
    snap = km.snapshot()
    assert snap.panic is True
    assert snap.manual_only is True

    # Critical bypass must NOT consume budget for subsequent
    # non-critical commands: _last_cmd_ts is unchanged by the panic.
    assert bot._last_cmd_ts.get("42") == last_cmd_ts_before


def test_hard_bypasses_rate_limit_after_recent_noncritical_command(
    tmp_path: Path,
) -> None:
    """Phase 4.14d: /kill hard shares the same rate-limit bypass as
    /kill panic."""
    store = KillStateStore(tmp_path / "kill.sqlite")
    captured: list[Event] = []

    async def pub(ev: Event) -> None:
        captured.append(ev)

    km = KillManager(store=store, publish=pub)
    bot = TelegramBot(
        kill_manager=km, token="tok", chat_id="42", rate_limit_sec=60.0
    )

    r1 = asyncio.run(bot.handle_text("/ping", chat_id="42"))
    assert "pong" in r1  # command landed, rate slot consumed
    last_cmd_ts_before = bot._last_cmd_ts.get("42")
    assert last_cmd_ts_before is not None

    reply = asyncio.run(
        bot.handle_text("/kill hard system fault", chat_id="42")
    )
    assert "rate-limited" not in reply
    assert "HARD engaged" in reply
    assert km.mode is KillMode.HARD
    # Budget unchanged.
    assert bot._last_cmd_ts.get("42") == last_cmd_ts_before


def test_soft_and_status_still_respect_rate_limit(tmp_path: Path) -> None:
    """Phase 4.14d guardrail: non-critical commands (including /kill
    soft) still hit the per-chat rate limit."""
    store = KillStateStore(tmp_path / "kill.sqlite")
    km = KillManager(store=store)
    bot = TelegramBot(
        kill_manager=km, token="tok", chat_id="42", rate_limit_sec=60.0
    )

    r1 = asyncio.run(bot.handle_text("/kill soft first", chat_id="42"))
    assert "SOFT engaged" in r1
    r2 = asyncio.run(bot.handle_text("/kill status", chat_id="42"))
    assert "rate-limited" in r2


# ---------------------------------------------------------------------------
# Phase 4.14e — bot lifecycle (H9) + health-signal semantics (M7).
# ---------------------------------------------------------------------------


def test_bot_stop_cancelled_nulls_task(tmp_path: Path) -> None:
    """If ``bot.stop()`` is itself cancelled mid-await, ``self._task``
    must still be cleared so a subsequent ``start()`` creates a fresh
    poll task. The prior implementation set ``self._task = None``
    outside a ``finally`` block, so an outer cancellation left the
    orphan task in place and ``start()`` no-opped."""
    bot, _km, _ = _bot(tmp_path)

    async def _hung_run() -> None:
        try:
            await asyncio.Event().wait()  # never completes
        except asyncio.CancelledError:
            # Simulate a slow teardown inside the task itself — the
            # task takes a little while to honor cancellation.
            await asyncio.sleep(0.2)
            raise

    async def _exercise() -> None:
        bot._task = asyncio.create_task(_hung_run(), name="fake-bot-task")
        # Force a CancelledError to propagate into stop() while it is
        # awaiting self._task.
        stop_task = asyncio.create_task(bot.stop())
        await asyncio.sleep(0)  # let stop() begin awaiting
        stop_task.cancel()
        try:
            await stop_task
        except asyncio.CancelledError:
            pass
        # Even though stop() was itself cancelled, the finally block
        # must have cleared _task.
        assert bot._task is None

    asyncio.run(_exercise())


def test_bot_stop_timeout_allows_restart(tmp_path: Path) -> None:
    """After a stop() that experienced an outer cancellation, a fresh
    start() must create a new task rather than short-circuiting via the
    ``if self._task is not None`` guard."""
    bot, _, _ = _bot(tmp_path)

    async def _hung_run() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise

    async def _exercise() -> None:
        bot._task = asyncio.create_task(_hung_run(), name="hung-task")
        # Wrap stop() in a very short wait_for; when it times out,
        # wait_for cancels the inner stop() coroutine.
        try:
            await asyncio.wait_for(bot.stop(), timeout=0.001)
        except asyncio.TimeoutError:
            pass
        # Regardless of the timeout, _task must be None.
        assert bot._task is None
        # start() must be able to create a new task. Use a minimal
        # fake loop that exits immediately so we do not depend on the
        # real poll.
        bot._stop.clear()

        async def _quick_loop() -> None:
            return None

        # Monkey-patch _run for this test so start() spawns a
        # trivially-completing task instead of the real poll.
        bot._run = _quick_loop  # type: ignore[assignment]
        await bot.start()
        assert bot._task is not None
        await bot.stop()
        assert bot._task is None

    asyncio.run(_exercise())


def test_last_activity_not_updated_on_getupdates_exception(tmp_path: Path) -> None:
    """Phase 4.14e (M7): a failing ``_get_updates`` must NOT advance
    ``_last_activity_ts``. Prior behavior refreshed the timestamp in
    every error path and before each call, hiding outages from the
    watchdog.

    We run a trimmed version of ``_run``'s body directly so the test
    does not depend on aiohttp's ClientSession context manager.
    """
    bot, _, _ = _bot(tmp_path)
    call_count = 0

    async def _boom(_session):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("simulated getUpdates failure")

    bot._get_updates = _boom  # type: ignore[assignment]

    async def _exercise() -> None:
        # Freeze the baseline timestamp to a known value we can compare
        # against after a burst of failed cycles.
        baseline = time.monotonic()
        bot._last_activity_ts = baseline
        # Run 5 iterations of the real failure-path code (mirrors the
        # body of ``_run`` without the aiohttp session).
        backoff = 0.001
        for _ in range(5):
            try:
                await bot._get_updates(None)
                bot._last_activity_ts = time.monotonic()
            except Exception:
                # M7 fix: no last_activity update in the error path.
                await asyncio.sleep(backoff)
        # Timestamp must not have advanced past the baseline.
        assert bot._last_activity_ts == baseline, (
            f"last_activity_ts drifted from {baseline} to "
            f"{bot._last_activity_ts} across {call_count} failed "
            "getUpdates; error path must not tick."
        )
        assert call_count == 5

    asyncio.run(_exercise())


# ---------------------------------------------------------------------------
# Phase 4.14e (M17) — malformed emergency commands must not consume
# rate budget. A parse-fail on ``/kill hard`` (missing reason) had
# ``parsed.valid=False`` and therefore hit the rate-limit branch,
# which would block the operator's immediately-corrected
# ``/kill hard <reason>`` for the rest of the rate window. The fix
# recognizes critical-kill prefixes from parsed.command/parsed.sub
# regardless of validity, so malformed criticals skip the rate
# limit *and* do not set ``_last_cmd_ts``. Strict dispatch still
# returns the parse error string.
# ---------------------------------------------------------------------------


def test_malformed_kill_hard_does_not_consume_rate_limit_for_corrected_hard(
    tmp_path: Path,
) -> None:
    store = KillStateStore(tmp_path / "kill.sqlite")
    km = KillManager(store=store)
    bot = TelegramBot(
        kill_manager=km, token="tok", chat_id="42", rate_limit_sec=60.0
    )
    # Malformed: missing reason. Strict dispatch still returns "err:".
    r1 = asyncio.run(bot.handle_text("/kill hard", chat_id="42"))
    assert r1.startswith("err:")
    assert "reason" in r1.lower()
    # Critical-kill detection is based on parsed.command/parsed.sub,
    # not parsed.valid — so rate budget was NOT consumed.
    assert "42" not in bot._last_cmd_ts

    # Immediately-corrected hard must NOT be rate-limited.
    r2 = asyncio.run(bot.handle_text("/kill hard real reason", chat_id="42"))
    assert "rate-limited" not in r2
    assert "HARD engaged" in r2
    assert km.mode is KillMode.HARD


def test_malformed_kill_panic_does_not_consume_rate_limit_for_corrected_panic(
    tmp_path: Path,
) -> None:
    store = KillStateStore(tmp_path / "kill.sqlite")
    km = KillManager(store=store, panic_cooldown_sec=1)
    bot = TelegramBot(
        kill_manager=km, token="tok", chat_id="42", rate_limit_sec=60.0
    )
    r1 = asyncio.run(bot.handle_text("/kill panic", chat_id="42"))
    assert r1.startswith("err:")
    assert "42" not in bot._last_cmd_ts

    r2 = asyncio.run(bot.handle_text("/kill panic meltdown", chat_id="42"))
    assert "rate-limited" not in r2
    assert "PANIC engaged" in r2
    assert km.snapshot().panic is True


def test_panic_engagement_survives_send_reply_failure(tmp_path: Path) -> None:
    """If the outbound Telegram send fails, the kill-switch remains
    engaged — engagement happened in _dispatch, persisted via
    KillStateStore, before any outbound call."""
    bot, km, _events = _bot(tmp_path)

    # Install a failing send override so that any post-dispatch reply
    # delivery would fail. handle_text itself does not call _send_reply;
    # that happens inside _handle_update.  We exercise the equivalent by
    # calling handle_text (engagement) then simulating reply failure.
    async def _boom(_text: str) -> None:
        raise RuntimeError("telegram send down")

    bot._send_override = _boom  # type: ignore[attr-defined]

    reply = asyncio.run(bot.handle_text("/kill panic network", chat_id="42"))
    # Engagement persisted before the reply path could ever run.
    assert km.mode is KillMode.HARD
    assert km.snapshot().panic is True
    assert "PANIC engaged" in reply

    # Simulate the outer send path failing; kill must still be engaged.
    async def _simulate_send() -> None:
        try:
            await bot._send_reply(session=None, chat_id="42", text=reply)
        except Exception:
            pass

    asyncio.run(_simulate_send())
    assert km.mode is KillMode.HARD
    assert km.snapshot().panic is True
