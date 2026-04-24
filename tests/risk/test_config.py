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


# --------------------------------------------------------------------------
# Phase 4.13 — range validation on numeric config fields (Finding #6)
# --------------------------------------------------------------------------


def test_config_rejects_negative_window_sec(tmp_path: Path):
    """Negative poisoning.window_sec must raise ConfigError at parse time."""
    p = tmp_path / "risk.yaml"
    p.write_text(yaml.safe_dump({"poisoning": {"window_sec": -5}}))
    with pytest.raises(ConfigError, match="poisoning.window_sec"):
        load_config(p)


def test_config_rejects_zero_min_samples(tmp_path: Path):
    """Zero poisoning.min_samples would disable the detector — reject."""
    p = tmp_path / "risk.yaml"
    p.write_text(yaml.safe_dump({"poisoning": {"min_samples": 0}}))
    with pytest.raises(ConfigError, match="poisoning.min_samples"):
        load_config(p)


def test_config_rejects_out_of_range_z_threshold(tmp_path: Path):
    """z_threshold must stay in [0.5, 20.0]: below = too-sensitive,
    above = effectively disabled."""
    p_low = tmp_path / "low.yaml"
    p_low.write_text(yaml.safe_dump({"poisoning": {"z_threshold": 0.1}}))
    with pytest.raises(ConfigError, match="poisoning.z_threshold"):
        load_config(p_low)

    p_high = tmp_path / "high.yaml"
    p_high.write_text(yaml.safe_dump({"poisoning": {"z_threshold": 100.0}}))
    with pytest.raises(ConfigError, match="poisoning.z_threshold"):
        load_config(p_high)


def test_config_rejects_negative_panic_cooldown(tmp_path: Path):
    """kill_switch.panic_cooldown_sec must be >= 0."""
    p = tmp_path / "risk.yaml"
    p.write_text(yaml.safe_dump({"kill_switch": {"panic_cooldown_sec": -1}}))
    with pytest.raises(ConfigError, match="kill_switch.panic_cooldown_sec"):
        load_config(p)


def test_config_accepts_valid_ranges_unchanged(tmp_path: Path):
    """Smoke test: known-valid values still load with no regression on
    default behavior or parsed values."""
    p = tmp_path / "risk.yaml"
    p.write_text(yaml.safe_dump({
        "poisoning": {
            "window_sec": 1800,
            "z_threshold": 3.5,
            "min_samples": 50,
            "pause_sec": 600,
        },
        "kill_switch": {
            "auto_resume_strike_limit": 5,
            "panic_cooldown_sec": 0,  # non-negative OK
        },
        "venue_health": {
            "window_sec": 30,
            "trip_threshold": 3,
            "pause_sec": 60,
        },
    }))
    cfg = load_config(p)
    assert cfg.poisoning.window_sec == 1800
    assert cfg.poisoning.z_threshold == 3.5
    assert cfg.poisoning.min_samples == 50
    assert cfg.kill_switch.panic_cooldown_sec == 0
    assert cfg.venue_health.window_sec == 30


# ---------------------------------------------------------------------------
# Phase 4.14b — DeadManCfg parsing.
# ---------------------------------------------------------------------------


def test_dead_man_disabled_by_default():
    cfg = load_config(None)
    assert cfg.dead_man.enabled is False
    assert cfg.dead_man.default_timeout_sec == 21600
    assert cfg.dead_man.min_timeout_sec == 300
    assert cfg.dead_man.max_timeout_sec == 43200


def test_dead_man_cfg_defaults_parse(tmp_path: Path):
    p = tmp_path / "risk.yaml"
    p.write_text(yaml.safe_dump({
        "dead_man": {
            "enabled": True,
            "default_timeout_sec": 1800,
            "min_timeout_sec": 60,
            "max_timeout_sec": 7200,
        }
    }))
    cfg = load_config(p)
    assert cfg.dead_man.enabled is True
    assert cfg.dead_man.default_timeout_sec == 1800
    assert cfg.dead_man.min_timeout_sec == 60
    assert cfg.dead_man.max_timeout_sec == 7200


def test_dead_man_cfg_invalid_bounds_rejected(tmp_path: Path):
    # min > default
    p1 = tmp_path / "bad1.yaml"
    p1.write_text(yaml.safe_dump({
        "dead_man": {
            "enabled": True,
            "default_timeout_sec": 60,
            "min_timeout_sec": 120,
            "max_timeout_sec": 600,
        }
    }))
    with pytest.raises(ConfigError):
        load_config(p1)

    # max < default
    p2 = tmp_path / "bad2.yaml"
    p2.write_text(yaml.safe_dump({
        "dead_man": {
            "enabled": True,
            "default_timeout_sec": 1800,
            "min_timeout_sec": 60,
            "max_timeout_sec": 600,
        }
    }))
    with pytest.raises(ConfigError):
        load_config(p2)

    # Negative / zero rejected by _require_positive_int
    p3 = tmp_path / "bad3.yaml"
    p3.write_text(yaml.safe_dump({
        "dead_man": {
            "enabled": True,
            "default_timeout_sec": 600,
            "min_timeout_sec": 0,
            "max_timeout_sec": 3600,
        }
    }))
    with pytest.raises(ConfigError):
        load_config(p3)
