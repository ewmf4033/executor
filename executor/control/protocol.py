"""
NDJSON control-plane protocol codec.

Wire format: newline-delimited JSON, one request per connection.

Request shape:
    {"cmd": str, "args": dict, "source": str}

Response (success):
    {"ok": true, "cmd": str, "result": dict}

Response (error):
    {"ok": false, "cmd": str, "error": str, "code": str}

Error codes: "unknown_command", "invalid_args", "kill_manager_error",
"rate_limited", "internal_error".

This module is codec-only — no network or filesystem IO.
"""
from __future__ import annotations

import json
from typing import Any


PROTOCOL_VERSION = 1

# Registered commands. Phase 4.14a: ping/version/kill_status/kill.
# Phase 4.14b adds arm/disarm/heartbeat/arm_status for the dead-man gate.
COMMANDS: frozenset[str] = frozenset(
    {
        "ping",
        "version",
        "kill_status",
        "kill",
        # Phase 4.14b — dead-man (operator liveness) control plane.
        "arm",
        "disarm",
        "heartbeat",
        "arm_status",
    }
)


class ProtocolError(Exception):
    """Malformed request, unknown command, or invalid arguments."""


def encode(obj: dict[str, Any]) -> bytes:
    """Serialize a response/request dict to an NDJSON line (trailing \\n)."""
    return (json.dumps(obj, default=str) + "\n").encode("utf-8")


def decode_request(raw: bytes) -> dict[str, Any]:
    """Parse one NDJSON request line. Returns a dict with keys
    ``cmd`` (str), ``args`` (dict), ``source`` (str).

    Raises ProtocolError on malformed JSON or invalid top-level shape.
    """
    try:
        text = raw.decode("utf-8").rstrip("\n")
        data = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError("request must be a JSON object")
    cmd = data.get("cmd")
    if not isinstance(cmd, str) or not cmd:
        raise ProtocolError("missing or invalid 'cmd' field")
    args = data.get("args", {})
    if not isinstance(args, dict):
        raise ProtocolError("'args' must be a JSON object")
    source = data.get("source", "unknown")
    if not isinstance(source, str):
        source = "unknown"
    return {"cmd": cmd, "args": args, "source": source}


def make_ok(cmd: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "cmd": cmd, "result": result}


def make_err(cmd: str, error: str, code: str) -> dict[str, Any]:
    return {"ok": False, "cmd": cmd, "error": error, "code": code}
