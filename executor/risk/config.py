"""
Risk config: conservative code defaults, YAML override, SIGHUP reload.

Every reload emits CONFIG_RELOADED with hash + timestamp (caller wires this
via ConfigManager.on_reload hooks; RiskPolicy registers the hook at __init__).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import time
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from ..core.logging import get_logger


log = get_logger("executor.risk.config")


# ---------------------------------------------------------------------------
# Conservative defaults. Intentionally small — promoting to real capital
# requires editing risk.yaml and reloading.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StructuralCfg:
    min_edge_default: Decimal = Decimal("0.01")
    per_strategy_min_edge: dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VenueHealthCfg:
    window_sec: int = 60
    trip_threshold: int = 2        # VenueDown/RateLimited incidents in window
    pause_sec: int = 120


@dataclass(frozen=True, slots=True)
class PerIntentCfg:
    max_intent_dollars: Decimal = Decimal("250.00")


@dataclass(frozen=True, slots=True)
class LiquidityCfg:
    depth_levels: int = 3
    min_remainder_contracts: Decimal = Decimal("1")


@dataclass(frozen=True, slots=True)
class ExposureCfg:
    default_ceiling_dollars: Decimal
    per_key: dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DailyLossCfg:
    default_max_loss_dollars: Decimal = Decimal("200.00")
    per_strategy: dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClipFloorCfg:
    min_final_ratio: Decimal = Decimal("0.5")


@dataclass(frozen=True, slots=True)
class PoisoningCfg:
    enabled: bool = True
    detector: str = "zscore"
    window_sec: int = 3600
    z_threshold: float = 5.0
    pause_sec: int = 300
    min_samples: int = 20
    detector_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdverseSelectionCfg:
    window: int = 20
    adverse_threshold: float = 0.60
    move_threshold_sigma: float = 2.0
    pause_sec: int = 300


@dataclass(frozen=True, slots=True)
class KillSwitchCfg:
    auto_resume_strike_limit: int = 3
    panic_cooldown_sec: int = 300


@dataclass(frozen=True, slots=True)
class AttributionCfg:
    exit_horizon_sec: int = 300


@dataclass(frozen=True, slots=True)
class RiskConfig:
    # Phase 4.13: when True, safety gates fail-closed rather than no-op when
    # their backing detector/tracker is unavailable. Default False preserves
    # paper-mode behavior for tests that disable subsystems.
    capital_mode: bool = False
    structural: StructuralCfg = field(default_factory=StructuralCfg)
    venue_health: VenueHealthCfg = field(default_factory=VenueHealthCfg)
    per_intent: PerIntentCfg = field(default_factory=PerIntentCfg)
    liquidity: LiquidityCfg = field(default_factory=LiquidityCfg)
    market_exposure: ExposureCfg = field(
        default_factory=lambda: ExposureCfg(default_ceiling_dollars=Decimal("500.00"))
    )
    event_concentration: ExposureCfg = field(
        default_factory=lambda: ExposureCfg(default_ceiling_dollars=Decimal("1000.00"))
    )
    venue_exposure: ExposureCfg = field(
        default_factory=lambda: ExposureCfg(default_ceiling_dollars=Decimal("2500.00"))
    )
    global_portfolio_dollars: Decimal = Decimal("10000.00")
    strategy_allocation: ExposureCfg = field(
        default_factory=lambda: ExposureCfg(default_ceiling_dollars=Decimal("1000.00"))
    )
    daily_loss: DailyLossCfg = field(default_factory=DailyLossCfg)
    clip_floor: ClipFloorCfg = field(default_factory=ClipFloorCfg)
    poisoning: PoisoningCfg = field(default_factory=PoisoningCfg)
    adverse_selection: AdverseSelectionCfg = field(default_factory=AdverseSelectionCfg)
    kill_switch: KillSwitchCfg = field(default_factory=KillSwitchCfg)
    attribution: AttributionCfg = field(default_factory=AttributionCfg)

    def fingerprint(self) -> str:
        """Stable SHA256 of the config; used for CONFIG_RELOADED payload."""
        serial = json.dumps(_to_jsonable(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serial.encode("utf-8")).hexdigest()


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    return obj


# ---------------------------------------------------------------------------
# YAML loader — merges overrides on top of defaults. Strict on type mismatch.
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised on malformed risk.yaml."""


def _require_positive_int(value: Any, field_name: str, *, max_val: int | None = None) -> int:
    """Coerce to int; require value > 0; optional upper bound.

    Raises ConfigError on type failure, non-positive value, or over-max.
    """
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer, got {value!r}") from exc
    if v <= 0:
        raise ConfigError(f"{field_name} must be positive, got {v}")
    if max_val is not None and v > max_val:
        raise ConfigError(f"{field_name} must be <= {max_val}, got {v}")
    return v


def _require_non_negative_int(value: Any, field_name: str, *, max_val: int | None = None) -> int:
    """Coerce to int; require value >= 0; optional upper bound."""
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer, got {value!r}") from exc
    if v < 0:
        raise ConfigError(f"{field_name} must be non-negative, got {v}")
    if max_val is not None and v > max_val:
        raise ConfigError(f"{field_name} must be <= {max_val}, got {v}")
    return v


def _require_range_float(value: Any, field_name: str, min_val: float, max_val: float) -> float:
    """Coerce to float; require min_val <= value <= max_val."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a float, got {value!r}") from exc
    if not (min_val <= v <= max_val):
        raise ConfigError(
            f"{field_name} must be in [{min_val}, {max_val}], got {v}"
        )
    return v


def load_config(path: str | os.PathLike[str] | None = None) -> RiskConfig:
    """Load RiskConfig from YAML at `path`. Returns defaults if path is None/missing."""
    if path is None:
        log.info("risk.config.defaults")
        return RiskConfig()
    p = Path(path)
    if not p.exists():
        log.info("risk.config.missing_file_using_defaults", path=str(p))
        return RiskConfig()
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"risk.yaml must be a mapping at the top level, got {type(raw).__name__}")

    try:
        cfg = RiskConfig(
            capital_mode=bool(raw.get("capital_mode", False)),
            structural=_parse_structural(raw.get("structural", {})),
            venue_health=_parse_venue_health(raw.get("venue_health", {})),
            per_intent=_parse_per_intent(raw.get("per_intent", {})),
            liquidity=_parse_liquidity(raw.get("liquidity", {})),
            market_exposure=_parse_exposure(raw.get("market_exposure", {}), Decimal("500.00")),
            event_concentration=_parse_exposure(raw.get("event_concentration", {}), Decimal("1000.00")),
            venue_exposure=_parse_exposure(raw.get("venue_exposure", {}), Decimal("2500.00")),
            global_portfolio_dollars=_dec(raw.get("global_portfolio_dollars", "10000.00")),
            strategy_allocation=_parse_exposure(raw.get("strategy_allocation", {}), Decimal("1000.00")),
            daily_loss=_parse_daily_loss(raw.get("daily_loss", {})),
            clip_floor=_parse_clip_floor(raw.get("clip_floor", {})),
            poisoning=_parse_poisoning(raw.get("poisoning", {})),
            adverse_selection=_parse_adverse_selection(raw.get("adverse_selection", {})),
            kill_switch=_parse_kill_switch(raw.get("kill_switch", {})),
            attribution=_parse_attribution(raw.get("attribution", {})),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid risk.yaml: {exc}") from exc
    log.info("risk.config.loaded", path=str(p), fingerprint=cfg.fingerprint())
    return cfg


def _dec(v: Any) -> Decimal:
    return Decimal(str(v))


def _parse_structural(d: dict[str, Any]) -> StructuralCfg:
    per = {str(k): _dec(v) for k, v in (d.get("per_strategy_min_edge") or {}).items()}
    return StructuralCfg(
        min_edge_default=_dec(d.get("min_edge_default", "0.01")),
        per_strategy_min_edge=per,
    )


def _parse_venue_health(d: dict[str, Any]) -> VenueHealthCfg:
    return VenueHealthCfg(
        window_sec=_require_positive_int(d.get("window_sec", 60), "venue_health.window_sec", max_val=86400),
        trip_threshold=_require_positive_int(d.get("trip_threshold", 2), "venue_health.trip_threshold", max_val=10000),
        pause_sec=_require_positive_int(d.get("pause_sec", 120), "venue_health.pause_sec", max_val=86400),
    )


def _parse_per_intent(d: dict[str, Any]) -> PerIntentCfg:
    return PerIntentCfg(max_intent_dollars=_dec(d.get("max_intent_dollars", "250.00")))


def _parse_liquidity(d: dict[str, Any]) -> LiquidityCfg:
    return LiquidityCfg(
        depth_levels=int(d.get("depth_levels", 3)),
        min_remainder_contracts=_dec(d.get("min_remainder_contracts", "1")),
    )


def _parse_exposure(d: dict[str, Any], default: Decimal) -> ExposureCfg:
    per_key = {str(k): _dec(v) for k, v in (d.get("per_key") or {}).items()}
    return ExposureCfg(
        default_ceiling_dollars=_dec(d.get("default_ceiling_dollars", str(default))),
        per_key=per_key,
    )


def _parse_daily_loss(d: dict[str, Any]) -> DailyLossCfg:
    per = {str(k): _dec(v) for k, v in (d.get("per_strategy") or {}).items()}
    return DailyLossCfg(
        default_max_loss_dollars=_dec(d.get("default_max_loss_dollars", "200.00")),
        per_strategy=per,
    )


def _parse_clip_floor(d: dict[str, Any]) -> ClipFloorCfg:
    return ClipFloorCfg(min_final_ratio=_dec(d.get("min_final_ratio", "0.5")))


def _parse_poisoning(d: dict[str, Any]) -> PoisoningCfg:
    return PoisoningCfg(
        enabled=bool(d.get("enabled", True)),
        detector=str(d.get("detector", "zscore")),
        window_sec=_require_positive_int(d.get("window_sec", 3600), "poisoning.window_sec", max_val=86400),
        z_threshold=_require_range_float(d.get("z_threshold", 5.0), "poisoning.z_threshold", 0.5, 20.0),
        pause_sec=_require_positive_int(d.get("pause_sec", 300), "poisoning.pause_sec", max_val=86400),
        min_samples=_require_positive_int(d.get("min_samples", 20), "poisoning.min_samples", max_val=10000),
        detector_kwargs=dict(d.get("detector_kwargs", {}) or {}),
    )


def _parse_adverse_selection(d: dict[str, Any]) -> AdverseSelectionCfg:
    return AdverseSelectionCfg(
        window=int(d.get("window", 20)),
        adverse_threshold=float(d.get("adverse_threshold", 0.60)),
        move_threshold_sigma=float(d.get("move_threshold_sigma", 2.0)),
        pause_sec=int(d.get("pause_sec", 300)),
    )


def _parse_kill_switch(d: dict[str, Any]) -> KillSwitchCfg:
    return KillSwitchCfg(
        auto_resume_strike_limit=_require_positive_int(
            d.get("auto_resume_strike_limit", 3),
            "kill_switch.auto_resume_strike_limit",
            max_val=100,
        ),
        panic_cooldown_sec=_require_non_negative_int(
            d.get("panic_cooldown_sec", 300),
            "kill_switch.panic_cooldown_sec",
            max_val=86400,
        ),
    )


def _parse_attribution(d: dict[str, Any]) -> AttributionCfg:
    return AttributionCfg(exit_horizon_sec=int(d.get("exit_horizon_sec", 300)))


# ---------------------------------------------------------------------------
# ConfigManager — holds current RiskConfig, reloads on SIGHUP, notifies hooks.
# ---------------------------------------------------------------------------


ReloadHook = Callable[[RiskConfig], Awaitable[None]]


class ConfigManager:
    """
    Owns the active RiskConfig. Reloads on SIGHUP or .reload() call.
    Every successful reload fires registered hooks (e.g. RiskPolicy
    emitting CONFIG_RELOADED, ZScoreDetector rebuilding its window).
    Invalid YAML is rejected — current config stays in place; hook is not called.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path = Path(path) if path else None
        self._config = load_config(self._path)
        self._hooks: list[ReloadHook] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._installed_signal = False
        self._last_reload_ts_ns: int = time.time_ns()

    @property
    def config(self) -> RiskConfig:
        return self._config

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def last_reload_ts_ns(self) -> int:
        return self._last_reload_ts_ns

    def register_reload_hook(self, hook: ReloadHook) -> None:
        self._hooks.append(hook)

    def install_sighup(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Register SIGHUP -> async reload. Idempotent. Best-effort on platforms
        without add_signal_handler (e.g. Windows)."""
        if self._installed_signal:
            return
        self._loop = loop or asyncio.get_event_loop()
        try:
            self._loop.add_signal_handler(signal.SIGHUP, self._on_sighup)
            self._installed_signal = True
            log.info("risk.config.sighup_installed")
        except (NotImplementedError, AttributeError):
            log.warning("risk.config.sighup_unsupported")

    def _on_sighup(self) -> None:
        assert self._loop is not None
        self._loop.create_task(self.reload(), name="risk-config-reload")

    async def reload(self) -> RiskConfig:
        """Re-read YAML and fire hooks. Raises ConfigError on malformed file;
        current config is preserved."""
        new_cfg = load_config(self._path)  # raises ConfigError if bad
        old_fp = self._config.fingerprint()
        self._config = new_cfg
        self._last_reload_ts_ns = time.time_ns()
        log.info(
            "risk.config.reloaded",
            path=str(self._path),
            old_fingerprint=old_fp,
            new_fingerprint=new_cfg.fingerprint(),
        )
        for hook in list(self._hooks):
            try:
                await hook(new_cfg)
            except Exception as exc:  # pragma: no cover — hook errors logged, config still active
                log.error("risk.config.reload_hook_error", error=str(exc))
        return new_cfg
