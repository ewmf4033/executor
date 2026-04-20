"""CLI safety gate (--paper / --live) tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from executor import cli


@pytest.fixture
def shipped_defaults_yaml(tmp_path: Path) -> Path:
    text = "structural:\n  min_edge_default: '0.01'\n"
    p = tmp_path / "risk.defaults.yaml"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def edited_yaml(tmp_path: Path) -> Path:
    text = "structural:\n  min_edge_default: '0.02'   # edited\n"
    p = tmp_path / "risk.yaml"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def matching_defaults_yaml(tmp_path: Path) -> Path:
    text = "structural:\n  min_edge_default: '0.01'\n"
    p = tmp_path / "risk.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_check_live_gate_missing_yaml() -> None:
    ok, why = cli.check_live_gate(risk_yaml_path="/nonexistent/risk.yaml", env={})
    assert ok is False
    assert "not found" in why


def test_check_live_gate_yaml_matches_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    text = "x: 1\n"
    (tmp_path / "risk.yaml").write_text(text)
    defaults = tmp_path / "risk.defaults.yaml"
    defaults.write_text(text)
    monkeypatch.setattr(cli, "_defaults_path", lambda: defaults)
    ok, why = cli.check_live_gate(
        risk_yaml_path=str(tmp_path / "risk.yaml"), env={"TELEGRAM_CHAT_ID": "x"}
    )
    assert ok is False
    assert "shipped defaults" in why


def test_check_live_gate_missing_chat_id(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "risk.yaml").write_text("a: 1\n")
    defaults = tmp_path / "risk.defaults.yaml"
    defaults.write_text("b: 2\n")
    monkeypatch.setattr(cli, "_defaults_path", lambda: defaults)
    ok, why = cli.check_live_gate(
        risk_yaml_path=str(tmp_path / "risk.yaml"), env={}
    )
    assert ok is False
    assert "TELEGRAM_CHAT_ID" in why


def test_check_live_gate_missing_ack(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "risk.yaml").write_text("a: 1\n")
    defaults = tmp_path / "risk.defaults.yaml"
    defaults.write_text("b: 2\n")
    monkeypatch.setattr(cli, "_defaults_path", lambda: defaults)
    ok, why = cli.check_live_gate(
        risk_yaml_path=str(tmp_path / "risk.yaml"),
        env={"TELEGRAM_CHAT_ID": "x"},
    )
    assert ok is False
    assert "EXECUTOR_LIVE_ACK" in why


def test_check_live_gate_wrong_ack(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "risk.yaml").write_text("a: 1\n")
    defaults = tmp_path / "risk.defaults.yaml"
    defaults.write_text("b: 2\n")
    monkeypatch.setattr(cli, "_defaults_path", lambda: defaults)
    ok, why = cli.check_live_gate(
        risk_yaml_path=str(tmp_path / "risk.yaml"),
        env={"TELEGRAM_CHAT_ID": "x", "EXECUTOR_LIVE_ACK": "wrong"},
    )
    assert ok is False


def test_check_live_gate_passes(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "risk.yaml").write_text("edited: yes\n")
    defaults = tmp_path / "risk.defaults.yaml"
    defaults.write_text("default: yes\n")
    monkeypatch.setattr(cli, "_defaults_path", lambda: defaults)
    ok, why = cli.check_live_gate(
        risk_yaml_path=str(tmp_path / "risk.yaml"),
        env={
            "TELEGRAM_CHAT_ID": "12345",
            "EXECUTOR_LIVE_ACK": cli.LIVE_ACK_PHRASE,
        },
    )
    assert ok is True, why
    assert why == ""


def test_resolve_mode_paper_pins_env(monkeypatch) -> None:
    monkeypatch.delenv("PAPER_MODE", raising=False)
    args = cli.parse_args(["--paper"])
    mode = cli.resolve_mode(args)
    assert mode == "paper"
    assert os.environ.get("PAPER_MODE") == "true"


def test_resolve_mode_default_is_paper(monkeypatch) -> None:
    monkeypatch.delenv("PAPER_MODE", raising=False)
    args = cli.parse_args([])
    mode = cli.resolve_mode(args)
    assert mode == "paper"
    assert os.environ["PAPER_MODE"] == "true"


def test_resolve_mode_live_exits_on_failed_gate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_defaults_path", lambda: tmp_path / "missing.yaml")
    args = cli.parse_args(["--live", "--risk-yaml", "/nope"])
    with pytest.raises(SystemExit) as ei:
        cli.resolve_mode(args, env={})
    assert ei.value.code == 2


def test_parse_args_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--paper", "--live"])
