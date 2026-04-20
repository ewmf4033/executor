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
