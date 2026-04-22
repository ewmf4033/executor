"""Phase 4.11.1 — TelegramBot daemon lifecycle wiring.

The bot reads TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID from os.environ at
__init__ and silently degrades (logs .disabled.no_token / .no_chat_id,
leaves _task as None) if either is missing. These tests exercise that
contract at the daemon-wiring level without touching Telegram servers.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from executor.kill.manager import KillManager
from executor.kill.state import KillStateStore
from executor.telegram.bot import TelegramBot


async def _noop_publish(_event):
    return None


def _make_kill_mgr(tmp_path: Path) -> KillManager:
    store = KillStateStore(tmp_path / "kill.sqlite")
    return KillManager(store=store, publish=_noop_publish)


async def test_telegram_bot_enabled_when_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env vars present → bot.start() spawns the poll task."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    kill_mgr = _make_kill_mgr(tmp_path)
    bot = TelegramBot(kill_manager=kill_mgr)
    # Prevent the poll loop from actually talking to Telegram: the _run
    # coroutine is replaced with one that blocks on a cancellable sleep.
    async def _stub_run(self) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    bot._run = _stub_run.__get__(bot, TelegramBot)  # type: ignore[assignment]

    try:
        await bot.start()
        assert bot._task is not None, "bot should spawn its poll task when env is set"
        assert not bot._task.done()
    finally:
        await bot.stop()


async def test_telegram_bot_disabled_when_env_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env vars absent → bot.start() is a silent no-op; _task stays None."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    kill_mgr = _make_kill_mgr(tmp_path)
    bot = TelegramBot(kill_manager=kill_mgr)

    await bot.start()
    assert bot._task is None, "bot must silently disable when env is missing"
    # stop() on a disabled bot must also be a clean no-op.
    await bot.stop()


async def test_telegram_bot_stop_clean_on_started_bot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stop() on a started bot tears the poll task down without raising."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-xyz")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "67890")

    kill_mgr = _make_kill_mgr(tmp_path)
    bot = TelegramBot(kill_manager=kill_mgr)

    async def _stub_run(self) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    bot._run = _stub_run.__get__(bot, TelegramBot)  # type: ignore[assignment]

    await bot.start()
    assert bot._task is not None

    # Must not raise.
    await bot.stop()
    assert bot._task is None
