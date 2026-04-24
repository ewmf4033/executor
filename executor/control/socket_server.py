"""
ControlSocketServer — AF_UNIX listener for operator control commands.

Authentication is filesystem permissions: the socket is created with
mode 0600 and the systemd unit runs as User=root (RuntimeDirectory=
executor gives us /run/executor/). Any process that can open(2) the
socket is trusted.

Commands routed through the already-wired KillManager so the audit
trail (KILL_COMMAND_RECEIVED with source="executorctl") and
fail-closed state machine are unchanged from the Telegram surface.

Rate limiting is deferred to Phase 4.14b — see the pause-before-commit
report for rationale. The per-connection protocol (one request per
connection) already prevents rapid-fire from a single client, the
socket is filesystem-permission-gated to root, and KillManager.engage
is idempotent for SOFT/HARD on the same mode, so a burst only rewrites
``reason`` + re-audits.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from ..kill.manager import KillManager
from ..kill.state import KillMode, KillStateSnapshot
from .protocol import (
    COMMANDS,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_request,
    encode,
    make_err,
    make_ok,
)


log = get_logger("executor.control.socket")


_READ_TIMEOUT_SEC = 2.0


try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except ImportError:  # pragma: no cover — 3.12+ always has it
    _pkg_version = None  # type: ignore[assignment]
    PackageNotFoundError = Exception  # type: ignore[misc,assignment]


def _get_executor_version() -> str:
    if _pkg_version is None:
        return "unknown"
    try:
        return _pkg_version("executor")
    except PackageNotFoundError:
        return "unknown"
    except Exception:
        return "unknown"


def _public_snapshot(snap: KillStateSnapshot) -> dict[str, Any]:
    """Project a KillStateSnapshot down to the operator-facing fields.

    We deliberately drop engaged_ts_ns / last_resume_ts_ns / panic_until_ns /
    extra so internal state stays out of the control-plane surface.
    """
    return {
        "mode": snap.mode.value,
        "panic": bool(snap.panic),
        "manual_only": bool(snap.manual_only),
        "strikes": int(snap.resume_strikes),
        "reason": snap.reason or "",
    }


class ControlSocketServer:
    def __init__(
        self,
        *,
        socket_path: str,
        kill_mgr: KillManager,
        daemon_started_ts_ns: int,
        git_sha: str | None = None,
    ) -> None:
        self._socket_path = str(socket_path)
        self._kill_mgr = kill_mgr
        self._daemon_started_ts_ns = int(daemon_started_ts_ns)
        self._git_sha = git_sha
        self._server: asyncio.base_events.Server | None = None

    async def start(self) -> None:
        path = Path(self._socket_path)
        # Parent directory is /run/executor (created by systemd
        # RuntimeDirectory) in production; create it for test/dev.
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        # Remove stale socket from a previous process.
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                log.warning(
                    "control.socket.stale_unlink_failed",
                    path=str(path),
                    error=str(exc),
                )
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=str(path)
        )
        try:
            os.chmod(str(path), 0o600)
        except OSError as exc:  # pragma: no cover
            log.warning(
                "control.socket.chmod_failed", path=str(path), error=str(exc)
            )
        log.info("control.socket.start", path=str(path))

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None
        try:
            p = Path(self._socket_path)
            if p.exists():
                p.unlink()
        except OSError:
            pass
        log.info("control.socket.stop", path=self._socket_path)

    # ------------------------------------------------------------------
    # Per-connection handler
    # ------------------------------------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        cmd_name = "?"
        try:
            try:
                raw = await asyncio.wait_for(
                    reader.readline(), timeout=_READ_TIMEOUT_SEC
                )
            except asyncio.TimeoutError:
                # Hung client — close silently without writing a response.
                return
            if not raw:
                return
            try:
                req = decode_request(raw)
            except ProtocolError as exc:
                writer.write(encode(make_err("?", str(exc), "invalid_args")))
                await writer.drain()
                return
            cmd_name = req["cmd"]
            args = req["args"]
            try:
                result = await self._dispatch(cmd_name, args)
            except ProtocolError as exc:
                msg = str(exc)
                code = (
                    "unknown_command"
                    if msg.startswith("unknown cmd")
                    else "invalid_args"
                )
                writer.write(encode(make_err(cmd_name, msg, code)))
                await writer.drain()
                return
            writer.write(encode(make_ok(cmd_name, result)))
            await writer.drain()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # One bad client must not crash the server.
            log.error(
                "control.socket.handler_error",
                cmd=cmd_name,
                error=str(exc),
            )
            try:
                writer.write(encode(make_err(cmd_name, "internal", "internal_error")))
                await writer.drain()
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, cmd: str, args: dict[str, Any]) -> dict[str, Any]:
        if cmd not in COMMANDS:
            raise ProtocolError(f"unknown cmd: {cmd}")
        if cmd == "ping":
            uptime_ns = max(0, time.time_ns() - self._daemon_started_ts_ns)
            return {
                "pong": True,
                "daemon_uptime_sec": int(uptime_ns // 1_000_000_000),
                "protocol_version": PROTOCOL_VERSION,
            }
        if cmd == "version":
            return {
                "executor_version": _get_executor_version(),
                "protocol_version": PROTOCOL_VERSION,
                "git_sha": self._git_sha,
            }
        if cmd == "kill_status":
            return _public_snapshot(self._kill_mgr.snapshot())
        if cmd == "kill":
            return await self._do_kill(args)
        # Unreachable given COMMANDS guard above.
        raise ProtocolError(f"unknown cmd: {cmd}")

    async def _do_kill(self, args: dict[str, Any]) -> dict[str, Any]:
        sub = args.get("sub")
        if sub not in ("soft", "hard", "panic", "resume"):
            raise ProtocolError(f"invalid sub: {sub!r}")
        reason = args.get("reason", "")
        if sub in ("soft", "hard", "panic"):
            if not isinstance(reason, str) or not reason:
                raise ProtocolError(
                    f"'reason' is required (non-empty string) for kill {sub}"
                )
        resume_blocked: str | None = None
        if sub == "soft":
            await self._kill_mgr.engage(
                KillMode.SOFT, reason, source="executorctl"
            )
        elif sub == "hard":
            await self._kill_mgr.engage(
                KillMode.HARD,
                reason,
                source="executorctl",
                cancel_open_orders=True,
            )
        elif sub == "panic":
            await self._kill_mgr.engage(
                KillMode.HARD,
                reason,
                source="executorctl",
                panic=True,
                cancel_open_orders=True,
            )
        else:  # resume
            ok, err = await self._kill_mgr.resume(source="executorctl")
            if not ok:
                resume_blocked = err
        result = _public_snapshot(self._kill_mgr.snapshot())
        result["source"] = "executorctl"
        if resume_blocked is not None:
            result["resume_blocked"] = resume_blocked
        return result
