"""Config loader — YAML override, SIGHUP reload, invalid YAML rejection."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from executor.risk.config import ConfigError, ConfigManager, RiskConfig, load_config


def test_defaults_when_no_path():
    cfg = load_config(None)
    assert cfg.per_intent.max_intent_dollars == Decimal("250.00")
    assert cfg.global_portfolio_dollars == Decimal("10000.00")
    assert cfg.poisoning.z_threshold == 5.0


def test_yaml_override_takes_precedence(tmp_path: Path):
    p = tmp_path / "risk.yaml"
    p.write_text(yaml.safe_dump({
        "per_intent": {"max_intent_dollars": "999.99"},
        "poisoning": {"z_threshold": 3.0, "enabled": False},
        "market_exposure": {"per_key": {"kalshi:X:YES": "123.45"}},
    }))
    cfg = load_config(p)
    assert cfg.per_intent.max_intent_dollars == Decimal("999.99")
    assert cfg.poisoning.z_threshold == 3.0
    assert cfg.poisoning.enabled is False
    assert cfg.market_exposure.per_key["kalshi:X:YES"] == Decimal("123.45")


def test_invalid_yaml_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("[: : ::: not yaml")
    with pytest.raises(ConfigError):
        load_config(p)


def test_non_mapping_yaml_raises(tmp_path: Path):
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_bad_field_type_raises(tmp_path: Path):
    p = tmp_path / "bad_field.yaml"
    p.write_text(yaml.safe_dump({"venue_health": {"window_sec": "not-an-int"}}))
    with pytest.raises(ConfigError):
        load_config(p)


@pytest.mark.asyncio
async def test_reload_invokes_hook_and_swaps_config(tmp_path: Path):
    p = tmp_path / "risk.yaml"
    p.write_text(yaml.safe_dump({"per_intent": {"max_intent_dollars": "100.00"}}))
    mgr = ConfigManager(p)
    assert mgr.config.per_intent.max_intent_dollars == Decimal("100.00")

    seen: list[RiskConfig] = []

    async def hook(cfg: RiskConfig) -> None:
        seen.append(cfg)

    mgr.register_reload_hook(hook)

    p.write_text(yaml.safe_dump({"per_intent": {"max_intent_dollars": "555.55"}}))
    await mgr.reload()
    assert mgr.config.per_intent.max_intent_dollars == Decimal("555.55")
    assert len(seen) == 1
    assert seen[0].per_intent.max_intent_dollars == Decimal("555.55")


@pytest.mark.asyncio
async def test_invalid_yaml_on_reload_preserves_current(tmp_path: Path):
    p = tmp_path / "risk.yaml"
    p.write_text(yaml.safe_dump({"per_intent": {"max_intent_dollars": "100.00"}}))
    mgr = ConfigManager(p)
    p.write_text("[: : garbage")
    with pytest.raises(ConfigError):
        await mgr.reload()
    # Old config still in place.
    assert mgr.config.per_intent.max_intent_dollars == Decimal("100.00")


def test_fingerprint_stable(tmp_path: Path):
    p = tmp_path / "risk.yaml"
    p.write_text(yaml.safe_dump({"per_intent": {"max_intent_dollars": "100.00"}}))
    a = load_config(p)
    b = load_config(p)
    assert a.fingerprint() == b.fingerprint()
