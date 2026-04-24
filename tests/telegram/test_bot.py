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
