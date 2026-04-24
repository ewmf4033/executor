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

from ..core.events import Event, EventType, Source
from ..core.logging import get_logger
from ..kill.manager import KillManager
from ..kill.state import KillMode, KillStateSnapshot
from ..risk.config import RiskConfig
from ..risk.state import OperatorLivenessStore
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
        # Phase 4.14b — optional operator-liveness wiring for the
        # dead-man gate. All three are optional so existing tests that
        # construct the server with only the kill surface still work;
        # dead-man commands that require any of them will return
        # "dead_man_disabled" / "internal_error" when absent.
        operator_liveness: OperatorLivenessStore | None = None,
        risk_config_getter: "Any | None" = None,
        publish: "Any | None" = None,
    ) -> None:
        self._socket_path = str(socket_path)
        self._kill_mgr = kill_mgr
        self._daemon_started_ts_ns = int(daemon_started_ts_ns)
        self._git_sha = git_sha
        self._operator_liveness = operator_liveness
        # Callable returning current RiskConfig (reads through to the
        # ConfigManager so SIGHUP reloads of dead_man bounds are picked
        # up without server restart). None => dead-man config not wired.
        self._risk_config_getter = risk_config_getter
        self._publish = publish
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
        # Phase 4.14b — operator liveness / dead-man.
        if cmd == "arm":
            return await self._do_arm(args)
        if cmd == "disarm":
            return await self._do_disarm(args)
        if cmd == "heartbeat":
            return await self._do_heartbeat(args)
        if cmd == "arm_status":
            return await self._do_arm_status(args)
        # Unreachable given COMMANDS guard above.
        raise ProtocolError(f"unknown cmd: {cmd}")

    # ------------------------------------------------------------------
    # Phase 4.14b — operator liveness / dead-man.
    # ------------------------------------------------------------------

    def _require_liveness(self) -> OperatorLivenessStore:
        if self._operator_liveness is None:
            raise ProtocolError(
                "dead_man_unavailable: operator_liveness store not wired"
            )
        return self._operator_liveness

    def _current_dead_man_cfg(self):
        """Return the current dead_man config section, or None if not wired.

        Read through the getter each call so SIGHUP reloads are picked up.
        """
        if self._risk_config_getter is None:
            return None
        cfg = self._risk_config_getter()
        return cfg.dead_man

    async def _publish_if_available(self, event: Event) -> None:
        if self._publish is None:
            return
        try:
            await self._publish(event)
        except Exception as exc:  # pragma: no cover — publish errors logged
            log.error("control.socket.publish_failed", error=str(exc))

    async def _do_arm(self, args: dict[str, Any]) -> dict[str, Any]:
        dm_cfg = self._current_dead_man_cfg()
        if dm_cfg is None:
            raise ProtocolError(
                "dead_man_unavailable: risk config not wired into control server"
            )
        if not dm_cfg.enabled:
            raise ProtocolError(
                "dead_man_disabled: cfg.dead_man.enabled is False; "
                "enable in risk.yaml before arming"
            )
        raw_timeout = args.get("timeout_sec")
        if not isinstance(raw_timeout, int) or isinstance(raw_timeout, bool):
            raise ProtocolError("'timeout_sec' must be an integer")
        if raw_timeout < dm_cfg.min_timeout_sec or raw_timeout > dm_cfg.max_timeout_sec:
            raise ProtocolError(
                f"'timeout_sec' out of bounds [{dm_cfg.min_timeout_sec},"
                f"{dm_cfg.max_timeout_sec}], got {raw_timeout}"
            )
        source = args.get("source", "executorctl")
        if not isinstance(source, str) or not source:
            source = "executorctl"
        store = self._require_liveness()
        kill_mode = self._kill_mgr.snapshot().mode.value
        now_ns = time.time_ns()
        store.arm(
            timeout_sec=raw_timeout,
            source=source,
            kill_mode=kill_mode,
            now_ns=now_ns,
        )
        await self._publish_if_available(
            Event.make(
                EventType.OPERATOR_ARMED,
                source=Source.EXECUTOR,
                payload={
                    "timeout_sec": int(raw_timeout),
                    "armed_by_source": source,
                    "kill_mode_at_arm": kill_mode,
                    "armed_ts_ns": int(now_ns),
                },
            )
        )
        return {
            "armed": True,
            "timeout_sec": int(raw_timeout),
            "kill_mode_at_arm": kill_mode,
            "last_heartbeat_ts_ns": int(now_ns),
        }

    async def _do_disarm(self, args: dict[str, Any]) -> dict[str, Any]:
        reason = args.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ProtocolError("'reason' is required (non-empty string) for disarm")
        store = self._require_liveness()
        now_ns = time.time_ns()
        store.disarm(reason=reason, now_ns=now_ns)
        await self._publish_if_available(
            Event.make(
                EventType.OPERATOR_DISARMED,
                source=Source.EXECUTOR,
                payload={"reason": reason, "disarmed_ts_ns": int(now_ns)},
            )
        )
        return {"armed": False, "disarmed_reason": reason}

    async def _do_heartbeat(self, args: dict[str, Any]) -> dict[str, Any]:
        store = self._require_liveness()
        now_ns = time.time_ns()
        was_armed = store.heartbeat(now_ns=now_ns)
        if not was_armed:
            return {
                "armed": False,
                "note": "heartbeat on disarmed state is a no-op",
            }
        snap = store.load()
        deadline_ns = snap.last_heartbeat_ts_ns + snap.timeout_sec * 1_000_000_000
        seconds_until_stale = int((deadline_ns - now_ns) / 1e9)
        await self._publish_if_available(
            Event.make(
                EventType.OPERATOR_HEARTBEAT,
                source=Source.EXECUTOR,
                payload={
                    "last_heartbeat_ts_ns": int(now_ns),
                    "seconds_until_stale": seconds_until_stale,
                },
            )
        )
        return {
            "armed": True,
            "last_heartbeat_ts_ns": int(now_ns),
            "seconds_until_stale": seconds_until_stale,
        }

    async def _do_arm_status(self, args: dict[str, Any]) -> dict[str, Any]:
        store = self._require_liveness()
        return store.status(now_ns=time.time_ns())

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
