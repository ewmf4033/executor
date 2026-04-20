"""
Telegram bot — async asyncio task inside the executor process.

Uses the same @masta_op2_bot token via env vars TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID. We poll Telegram's getUpdates endpoint over plain HTTPS
(aiohttp) so we don't pull in a third-party Telegram SDK.

Commands (Phase 4 surface):
  /kill soft <reason>
  /kill hard <reason>
  /kill panic <reason>     (HARD + disable auto-resume + 5-min cooldown)
  /kill resume             (clears kill if circuit allows)
  /kill status
  /venue health

Authorization:
- Only messages whose chat_id matches TELEGRAM_CHAT_ID are accepted.
- Other chats are silently ignored (and logged).

Rate limit:
- Max 1 command per 2 seconds per chat_id. Excess commands are dropped
  with a "rate-limited" reply.

Audit:
- KILL_COMMAND_RECEIVED is emitted via KillManager.emit_command_received
  for every accepted command (parse failures included for the audit trail).

The bot is robust to network errors — getUpdates failures sleep and retry.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

try:
    import aiohttp  # type: ignore
except ImportError:  # pragma: no cover — aiohttp is a hard dep declared in pyproject
    aiohttp = None  # type: ignore

from ..core.logging import get_logger
from ..kill.manager import KillManager
from ..kill.state import KillMode


log = get_logger("executor.telegram.bot")


TELEGRAM_API = "https://api.telegram.org"
DEFAULT_RATE_LIMIT_SEC = 2.0
DEFAULT_POLL_TIMEOUT_SEC = 25
DEFAULT_LOOP_BACKOFF_SEC = 1.0


@dataclass(frozen=True)
class ParsedCommand:
    valid: bool
    command: str            # "kill" | "venue" | ""
    sub: str                # "soft" | "hard" | "panic" | "resume" | "status" | "health" | ""
    args: str
    error: str = ""


def parse_command(text: str) -> ParsedCommand:
    """
    Strict parser for the Phase 4 surface. Returns valid=False on garbage.
    Strips an optional bot-mention suffix (e.g. "/kill@masta_op2_bot soft x").
    """
    raw = (text or "").strip()
    if not raw.startswith("/"):
        return ParsedCommand(False, "", "", "", "not a command")
    parts = raw[1:].split(maxsplit=2)
    if not parts:
        return ParsedCommand(False, "", "", "", "empty command")
    head = parts[0].split("@", 1)[0].lower()  # strip @bot mention
    if head not in ("kill", "venue"):
        return ParsedCommand(False, head, "", "", f"unknown command: {head}")
    sub = parts[1].lower() if len(parts) > 1 else ""
    args = parts[2] if len(parts) > 2 else ""

    if head == "kill":
        if sub not in ("soft", "hard", "panic", "resume", "status"):
            return ParsedCommand(False, head, sub, args, f"unknown /kill sub: {sub}")
        if sub in ("soft", "hard", "panic") and not args:
            return ParsedCommand(False, head, sub, args, f"/kill {sub} requires <reason>")
        return ParsedCommand(True, head, sub, args, "")
    if head == "venue":
        if sub != "health":
            return ParsedCommand(False, head, sub, args, f"unknown /venue sub: {sub}")
        return ParsedCommand(True, head, sub, args, "")
    return ParsedCommand(False, head, sub, args, "unhandled")


class TelegramBot:
    """Lifecycle: bot.start() -> bot.stop(). Owns one background asyncio task."""

    def __init__(
        self,
        *,
        kill_manager: KillManager,
        token: str | None = None,
        chat_id: str | int | None = None,
        rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
        venue_health_provider: Callable[[], Awaitable[dict[str, Any]] | dict[str, Any]] | None = None,
        send_callback: Callable[[str], Awaitable[None]] | None = None,
        poll_timeout_sec: int = DEFAULT_POLL_TIMEOUT_SEC,
    ) -> None:
        self._kill = kill_manager
        self._token = token if token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")
        env_chat = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")
        self._chat_id: str = str(env_chat) if env_chat not in (None, "") else ""
        self._rate_limit_sec = rate_limit_sec
        self._venue_health = venue_health_provider
        # send_callback overrides HTTP send (used in tests).
        self._send_override = send_callback
        self._poll_timeout_sec = poll_timeout_sec
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._last_cmd_ts: dict[str, float] = {}
        self._update_offset: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        if not self._token:
            log.warning("telegram.bot.disabled.no_token")
            return
        if not self._chat_id:
            log.warning("telegram.bot.disabled.no_chat_id")
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="telegram-bot")
        log.info("telegram.bot.start")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None
        log.info("telegram.bot.stop")

    # ------------------------------------------------------------------
    # Public command dispatch (also used directly by tests)
    # ------------------------------------------------------------------

    async def handle_text(self, text: str, *, chat_id: str | int) -> str:
        """
        Dispatch a single text message. Returns the reply that would be sent.
        Enforces chat_id auth + per-chat rate limit. Always emits an audit
        KILL_COMMAND_RECEIVED for accepted (non-rate-limited) commands.
        """
        chat_str = str(chat_id)
        if self._chat_id and chat_str != self._chat_id:
            log.warning("telegram.bot.unauthorized_chat", chat_id=chat_str)
            return ""

        now = time.monotonic()
        last = self._last_cmd_ts.get(chat_str, 0.0)
        if now - last < self._rate_limit_sec:
            wait = self._rate_limit_sec - (now - last)
            return f"rate-limited; wait {wait:.1f}s"
        self._last_cmd_ts[chat_str] = now

        parsed = parse_command(text)
        await self._kill.emit_command_received(
            command=parsed.command,
            args=f"{parsed.sub} {parsed.args}".strip(),
            chat_id=chat_str,
        )
        if not parsed.valid:
            return f"err: {parsed.error}"
        return await self._dispatch(parsed)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _dispatch(self, p: ParsedCommand) -> str:
        if p.command == "kill":
            if p.sub == "soft":
                snap = await self._kill.engage(KillMode.SOFT, p.args, source="telegram")
                return f"OK: SOFT engaged ({snap.reason})"
            if p.sub == "hard":
                snap = await self._kill.engage(KillMode.HARD, p.args, source="telegram")
                return f"OK: HARD engaged ({snap.reason})"
            if p.sub == "panic":
                snap = await self._kill.engage(
                    KillMode.HARD, p.args, source="telegram", panic=True
                )
                return f"OK: PANIC engaged ({snap.reason}); cooldown until ts_ns={snap.panic_until_ns}"
            if p.sub == "resume":
                ok, why = await self._kill.resume(source="telegram")
                return "OK: resumed" if ok else f"BLOCKED: {why}"
            if p.sub == "status":
                snap = self._kill.snapshot()
                return (
                    f"mode={snap.mode.value} panic={snap.panic} "
                    f"manual_only={snap.manual_only} strikes={snap.resume_strikes} "
                    f"reason={snap.reason!r}"
                )
        if p.command == "venue" and p.sub == "health":
            if self._venue_health is None:
                return "venue health: no provider attached"
            try:
                raw = self._venue_health()
                if asyncio.iscoroutine(raw):
                    raw = await raw  # type: ignore
            except Exception as exc:
                return f"venue health: error {exc}"
            return "venue health: " + ", ".join(
                f"{k}={v}" for k, v in sorted((raw or {}).items())
            )
        return "err: unhandled command"

    # ------------------------------------------------------------------
    # Long-running poll loop (real Telegram)
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        if aiohttp is None:  # pragma: no cover
            log.error("telegram.bot.run.no_aiohttp")
            return
        backoff = DEFAULT_LOOP_BACKOFF_SEC
        async with aiohttp.ClientSession() as session:  # type: ignore
            while not self._stop.is_set():
                try:
                    updates = await self._get_updates(session)
                    for upd in updates:
                        await self._handle_update(session, upd)
                    backoff = DEFAULT_LOOP_BACKOFF_SEC
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("telegram.bot.loop.error", error=str(exc))
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)

    async def _get_updates(self, session: Any) -> list[dict[str, Any]]:
        url = f"{TELEGRAM_API}/bot{self._token}/getUpdates"
        params = {"timeout": self._poll_timeout_sec, "offset": self._update_offset}
        async with session.get(url, params=params, timeout=self._poll_timeout_sec + 5) as resp:
            data = await resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"getUpdates failed: {data}")
        result = data.get("result") or []
        if result:
            self._update_offset = max(int(u.get("update_id", 0)) for u in result) + 1
        return result

    async def _handle_update(self, session: Any, upd: dict[str, Any]) -> None:
        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            return
        chat = (msg.get("chat") or {}).get("id")
        text = msg.get("text") or ""
        reply = await self.handle_text(text, chat_id=chat)
        if reply:
            await self._send_reply(session, chat, reply)

    async def _send_reply(self, session: Any, chat_id: Any, text: str) -> None:
        if self._send_override is not None:
            await self._send_override(text)
            return
        url = f"{TELEGRAM_API}/bot{self._token}/sendMessage"
        body = {"chat_id": chat_id, "text": text}
        try:
            async with session.post(url, json=body, timeout=10) as resp:
                if resp.status != 200:
                    log.warning("telegram.bot.send.bad_status", status=resp.status)
        except Exception as exc:
            log.warning("telegram.bot.send.error", error=str(exc))
