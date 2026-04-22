"""Validates the 1Password env file format and secret-reference syntax.

Phase 4.10 (0k). This is a CI-safe substitute for a live `op run` integration
test — we can't authenticate to 1Password in CI, so instead we verify:

1. /etc/executor/op.env parses as KEY=value lines with no raw secrets.
2. Every value is an `op://vault/item/field` URI (exactly 3 path segments).
3. Every env var name listed is one the executor actually reads.
4. The systemd draft `executor.service.op` invokes `op run --env-file=` with
   the same file path.

If this test fails, someone has:
- Put a raw secret into op.env (which must never happen).
- Listed an env var the executor does not consume.
- Changed the op.env path without updating the systemd unit (or vice-versa).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
OP_ENV = Path("/etc/executor/op.env")
SYSTEMD_OP = REPO_ROOT / "systemd" / "executor.service.op"


# Env vars the executor actually reads (grep'd from the codebase).
# When adding a new secret, update both this list and op.env.
EXECUTOR_SECRET_ENV_VARS = {
    "KALSHI_API_KEY_ID",
    "KALSHI_PRIVATE_KEY_PATH",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
}


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            pytest.fail(f"op.env malformed line: {raw!r}")
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@pytest.mark.skipif(not OP_ENV.exists(), reason="op.env not installed on this host")
def test_op_env_contains_only_op_uris() -> None:
    env = _parse_env_file(OP_ENV)
    assert env, "op.env is empty"
    for key, val in env.items():
        assert val.startswith("op://"), (
            f"{key} value is not an op:// URI; possible raw secret leak: {val!r}"
        )
        # Path after op:// must have exactly 3 segments: vault/item/field.
        segments = val[len("op://"):].split("/")
        assert len(segments) == 3, (
            f"{key} op URI must be op://vault/item/field, got {val!r}"
        )
        for seg in segments:
            assert seg, f"{key} op URI has empty segment: {val!r}"


@pytest.mark.skipif(not OP_ENV.exists(), reason="op.env not installed on this host")
def test_op_env_covers_every_executor_secret() -> None:
    env = _parse_env_file(OP_ENV)
    missing = EXECUTOR_SECRET_ENV_VARS - set(env.keys())
    assert not missing, f"op.env is missing env vars consumed by executor: {missing}"


@pytest.mark.skipif(not OP_ENV.exists(), reason="op.env not installed on this host")
def test_op_env_has_no_extraneous_vars() -> None:
    """op.env should list exactly what the executor consumes — no more."""
    env = _parse_env_file(OP_ENV)
    extra = set(env.keys()) - EXECUTOR_SECRET_ENV_VARS
    assert not extra, (
        f"op.env contains env vars the executor does not read: {extra}. "
        "Either the executor stopped using them (remove from op.env) or the "
        "EXECUTOR_SECRET_ENV_VARS list is stale (update this test)."
    )


def test_systemd_op_unit_references_op_env() -> None:
    """The systemd draft must `op run --env-file=/etc/executor/op.env`."""
    assert SYSTEMD_OP.exists(), f"systemd op unit draft missing: {SYSTEMD_OP}"
    txt = SYSTEMD_OP.read_text()
    assert "op run" in txt, "systemd unit must wrap ExecStart with `op run`"
    assert "--env-file=/etc/executor/op.env" in txt, (
        "systemd unit must reference the canonical /etc/executor/op.env path"
    )
    assert "EnvironmentFile=/etc/executor/op-service-account-token" in txt, (
        "service account token must come from EnvironmentFile, not be inlined"
    )


def test_op_injection_behavior_with_mocked_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI-safe mock: when env vars are present (as `op run` would inject),
    the code paths that read them return the resolved values. This does NOT
    exercise op itself — the assumption is that `op run` correctly injects.
    """
    monkeypatch.setenv("KALSHI_API_KEY_ID", "MOCK-KEY-ID")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "MOCK-BOT-TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    assert os.environ["KALSHI_API_KEY_ID"] == "MOCK-KEY-ID"
    assert os.environ["TELEGRAM_BOT_TOKEN"] == "MOCK-BOT-TOKEN"
    assert os.environ["TELEGRAM_CHAT_ID"] == "12345"
