"""Control-plane protocol codec — encode/decode + shape validation."""
from __future__ import annotations

import json

import pytest

from executor.control.protocol import (
    COMMANDS,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_request,
    encode,
    make_err,
    make_ok,
)


def test_commands_registry_and_version() -> None:
    # Phase 4.14b added arm/disarm/heartbeat/arm_status for the dead-man
    # gate control surface.
    assert COMMANDS == {
        "ping",
        "version",
        "kill_status",
        "kill",
        "arm",
        "disarm",
        "heartbeat",
        "arm_status",
    }
    assert PROTOCOL_VERSION == 1


def test_encode_decode_roundtrip() -> None:
    raw = encode(make_ok("ping", {"pong": True}))
    assert raw.endswith(b"\n")
    data = json.loads(raw.decode("utf-8"))
    assert data == {"ok": True, "cmd": "ping", "result": {"pong": True}}

    err = encode(make_err("x", "nope", "invalid_args"))
    assert json.loads(err.decode("utf-8")) == {
        "ok": False,
        "cmd": "x",
        "error": "nope",
        "code": "invalid_args",
    }


def test_decode_request_accepts_valid() -> None:
    req = decode_request(b'{"cmd": "ping", "args": {}, "source": "ctl"}\n')
    assert req == {"cmd": "ping", "args": {}, "source": "ctl"}


def test_decode_rejects_malformed_json() -> None:
    with pytest.raises(ProtocolError):
        decode_request(b"not valid json{")


def test_decode_rejects_missing_cmd() -> None:
    with pytest.raises(ProtocolError):
        decode_request(b'{"args": {}}')


def test_decode_rejects_non_string_cmd() -> None:
    with pytest.raises(ProtocolError):
        decode_request(b'{"cmd": 42, "args": {}}')


def test_decode_rejects_non_object_shape() -> None:
    with pytest.raises(ProtocolError):
        decode_request(b'["ping"]')


def test_decode_rejects_non_dict_args() -> None:
    with pytest.raises(ProtocolError):
        decode_request(b'{"cmd": "ping", "args": "oops"}')
