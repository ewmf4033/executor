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

from ..core.events import Event, EventType, Source
from ..core.logging import get_logger
from ..kill.manager import KillManager
from ..kill.state import KillMode
from ..risk.config import DeadManCfg
from ..risk.state import OperatorLivenessStore


log = get_logger("executor.telegram.bot")


TELEGRAM_API = "https://api.telegram.org"
DEFAULT_RATE_LIMIT_SEC = 2.0
DEFAULT_POLL_TIMEOUT_SEC = 25
DEFAULT_LOOP_BACKOFF_SEC = 1.0


def _parse_tg_timeout(raw: str) -> int:
    """Parse a Telegram-style duration token ("300", "5m", "6h") -> seconds.

    Rejects day suffix for T1 safety (mirrors executorctl's parser).
    """
    if not raw:
        raise ValueError("empty timeout")
    unit = raw[-1].lower()
    if unit.isdigit():
        return int(raw)
    if unit == "s":
        return int(raw[:-1])
    if unit == "m":
        return int(raw[:-1]) * 60
    if unit == "h":
        return int(raw[:-1]) * 3600
    raise ValueError(f"unsupported timeout unit {unit!r}")


def _fmt_sec(seconds: int) -> str:
    sign = "-" if seconds < 0 else ""
    s = abs(int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{sec}s")
    return sign + "".join(parts)


@dataclass(frozen=True)
class ParsedCommand:
    valid: bool
    command: str            # "kill" | "venue" | "ping" | "heartbeat" | "arm" | "disarm" | "arm_status" | ""
    sub: str                # "soft" | "hard" | "panic" | "resume" | "status" | "health" | ""
    args: str
    error: str = ""


# Phase 4.14b: /ping, /heartbeat, /arm_status take no arguments. /arm takes
# a single timeout token (sub). /disarm takes the rest of the line as its
# reason (args). /ping is explicitly distinct from /heartbeat — the former
# is status-only and does NOT update last_heartbeat_ts_ns.
_DEAD_MAN_NO_ARG = frozenset({"ping", "heartbeat", "arm_status"})


def parse_command(text: str) -> ParsedCommand:
    """
    Strict parser for the Phase 4 / 4.14b surface. Returns valid=False on garbage.
    Strips an optional bot-mention suffix (e.g. "/kill@masta_op2_bot soft x").
    """
    raw = (text or "").strip()
    if not raw.startswith("/"):
        return ParsedCommand(False, "", "", "", "not a command")
    parts = raw[1:].split(maxsplit=2)
    if not parts:
        return ParsedCommand(False, "", "", "", "empty command")
    head = parts[0].split("@", 1)[0].lower()  # strip @bot mention
    known = {"kill", "venue", "ping", "heartbeat", "arm", "disarm", "arm_status"}
    if head not in known:
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
    # Phase 4.14b — dead-man commands.
    if head in _DEAD_MAN_NO_ARG:
        # No arguments expected; tolerate extras by ignoring.
        return ParsedCommand(True, head, "", "", "")
    if head == "arm":
        if not sub:
            return ParsedCommand(False, head, sub, args, "/arm requires <timeout>")
        # Preserve original case of the timeout token.
        orig_sub = parts[1] if len(parts) > 1 else ""
        return ParsedCommand(True, head, orig_sub, "", "")
    if head == "disarm":
        # Join sub + args back into a single free-form reason string.
        reason = (parts[1] + (" " + args if args else "")) if len(parts) > 1 else ""
        if not reason:
            return ParsedCommand(False, head, "", "", "/disarm requires <reason>")
        return ParsedCommand(True, head, "", reason, "")
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
        # Phase 4.14b — optional dead-man wiring. None preserves the
        # Phase 4 / 4.11.1 behavior for existing tests: /arm /disarm /...
        # commands then reply with an "unavailable" error rather than
        # silently no-opping.
        operator_liveness: OperatorLivenessStore | None = None,
        dead_man_cfg_getter: Callable[[], DeadManCfg] | None = None,
        publish: Callable[[Event], Awaitable[None]] | None = None,
    ) -> None:
        self._kill = kill_manager
        self._operator_liveness = operator_liveness
        self._dead_man_cfg_getter = dead_man_cfg_getter
        self._publish = publish
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
        # Phase 4.14b — dead-man commands.
        if p.command == "ping":
            return self._fmt_ping()
        if p.command == "arm_status":
            return self._fmt_arm_status()
        if p.command == "heartbeat":
            return await self._do_heartbeat()
        if p.command == "arm":
            return await self._do_arm(p.sub)
        if p.command == "disarm":
            return await self._do_disarm(p.args)
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
    # Phase 4.14b — dead-man helpers.
    # ------------------------------------------------------------------

    def _fmt_ping(self) -> str:
        """Status-only pong; NEVER updates last_heartbeat_ts_ns."""
        snap = self._kill.snapshot()
        if self._operator_liveness is None:
            return f"pong | kill={snap.mode.value}"
        live = self._operator_liveness.load()
        if not live.armed:
            return f"pong | armed=False kill={snap.mode.value}"
        deadline = live.last_heartbeat_ts_ns + live.timeout_sec * 1_000_000_000
        until_stale = int((deadline - time.time_ns()) / 1e9)
        return (
            f"pong | armed=True until_stale={_fmt_sec(until_stale)} "
            f"kill={snap.mode.value}"
        )

    def _fmt_arm_status(self) -> str:
        if self._operator_liveness is None:
            return "arm_status: operator_liveness not wired"
        live = self._operator_liveness.load()
        if not live.armed:
            reason = f" last_disarm_reason={live.disarmed_reason!r}" if live.disarmed_reason else ""
            return f"armed=False{reason}"
        deadline = live.last_heartbeat_ts_ns + live.timeout_sec * 1_000_000_000
        until_stale = int((deadline - time.time_ns()) / 1e9)
        return (
            f"armed=True timeout={_fmt_sec(live.timeout_sec)} "
            f"until_stale={_fmt_sec(until_stale)} "
            f"kill_mode_at_arm={live.kill_mode_at_arm}"
        )

    async def _do_heartbeat(self) -> str:
        if self._operator_liveness is None:
            return "heartbeat: operator_liveness not wired"
        now_ns = time.time_ns()
        was_armed = self._operator_liveness.heartbeat(now_ns=now_ns)
        if not was_armed:
            return "not armed — heartbeat ignored"
        live = self._operator_liveness.load()
        deadline = live.last_heartbeat_ts_ns + live.timeout_sec * 1_000_000_000
        until_stale = int((deadline - now_ns) / 1e9)
        if self._publish is not None:
            try:
                await self._publish(
                    Event.make(
                        EventType.OPERATOR_HEARTBEAT,
                        source=Source.TELEGRAM,
                        payload={
                            "last_heartbeat_ts_ns": int(now_ns),
                            "seconds_until_stale": until_stale,
                        },
                    )
                )
            except Exception as exc:  # pragma: no cover
                log.warning("telegram.bot.heartbeat_publish_failed", error=str(exc))
        return f"heartbeat ok — {_fmt_sec(until_stale)} until stale"

    async def _do_arm(self, timeout_token: str) -> str:
        if self._operator_liveness is None or self._dead_man_cfg_getter is None:
            return "arm: dead-man not wired"
        cfg = self._dead_man_cfg_getter()
        if not cfg.enabled:
            return "arm: dead_man.enabled is False in risk.yaml"
        try:
            timeout_sec = _parse_tg_timeout(timeout_token)
        except ValueError as exc:
            return f"arm: invalid timeout ({exc})"
        if not (cfg.min_timeout_sec <= timeout_sec <= cfg.max_timeout_sec):
            return (
                f"arm: timeout_sec {timeout_sec} out of bounds "
                f"[{cfg.min_timeout_sec},{cfg.max_timeout_sec}]"
            )
        kill_mode = self._kill.snapshot().mode.value
        now_ns = time.time_ns()
        self._operator_liveness.arm(
            timeout_sec=timeout_sec,
            source="telegram",
            kill_mode=kill_mode,
            now_ns=now_ns,
        )
        if self._publish is not None:
            try:
                await self._publish(
                    Event.make(
                        EventType.OPERATOR_ARMED,
                        source=Source.TELEGRAM,
                        payload={
                            "timeout_sec": int(timeout_sec),
                            "armed_by_source": "telegram",
                            "kill_mode_at_arm": kill_mode,
                            "armed_ts_ns": int(now_ns),
                        },
                    )
                )
            except Exception as exc:  # pragma: no cover
                log.warning("telegram.bot.arm_publish_failed", error=str(exc))
        return f"armed for {_fmt_sec(timeout_sec)} | kill={kill_mode} at arm time"

    async def _do_disarm(self, reason: str) -> str:
        if self._operator_liveness is None:
            return "disarm: dead-man not wired"
        reason = (reason or "").strip()
        if not reason:
            return "disarm: reason required"
        now_ns = time.time_ns()
        self._operator_liveness.disarm(reason=reason, now_ns=now_ns)
        if self._publish is not None:
            try:
                await self._publish(
                    Event.make(
                        EventType.OPERATOR_DISARMED,
                        source=Source.TELEGRAM,
                        payload={"reason": reason, "disarmed_ts_ns": int(now_ns)},
                    )
                )
            except Exception as exc:  # pragma: no cover
                log.warning("telegram.bot.disarm_publish_failed", error=str(exc))
        return f"disarmed — reason: {reason}"

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
