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
class DeadManCfg:
    """Phase 4.14b — operator availability gate (Gate 8.5).

    When ``enabled`` is False the DeadManGate bypasses entirely (paper
    default). When True, the operator must explicitly arm the dead-man
    and provide heartbeats within ``timeout_sec``; stale or disarmed
    state blocks new intents.

    Bounds: 300s (5 min) <= min_timeout_sec <= default_timeout_sec
    <= max_timeout_sec <= 43200s (12h).
    """
    enabled: bool = False
    default_timeout_sec: int = 21600  # 6h
    min_timeout_sec: int = 300        # 5 min
    max_timeout_sec: int = 43200      # 12h


@dataclass(frozen=True, slots=True)
class FeeGateCfg:
    """Phase 4.15 — fee-aware executable-edge gate (slot 1.5).

    bps-of-notional is intentionally a placeholder; Phase 5a will replace
    the bps formula with a venue-native maker/taker fee estimator without
    restructuring the gate. Lookup precedence:
      1. per_market_fee_bps["venue:market_id"]
      2. per_series_fee_bps[prefix] (longest matching prefix on market_id)
      3. default_fee_bps
    """
    enabled: bool = True
    apply_in_paper_mode: bool = False
    default_fee_bps: Decimal = Decimal("0")
    safety_margin_bps: Decimal = Decimal("0")
    per_market_fee_bps: dict[str, Decimal] = field(default_factory=dict)
    per_series_fee_bps: dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrderPolicyCfg:
    """Phase 4.15 — order-policy gate (slot 1.6).

    Validates leg.metadata for venue-write shape. Paper mode is
    permissive about ABSENCE of metadata but strict about PRESENCE of
    explicitly unsafe metadata. Capital mode adds required-key checks.
    """
    enabled: bool = True
    apply_in_paper_mode: bool = False
    allowed_time_in_force: tuple[str, ...] = ("IOC", "FOK")
    forbid_post_only: bool = True
    forbid_reduce_only: bool = False
    require_order_group_id_in_capital_mode: bool = True
    require_buy_max_cost_for_buys_in_capital_mode: bool = True


@dataclass(frozen=True, slots=True)
class HostHealthCfg:
    """Phase 4.16 — host-health admission gate.

    Disabled by default. Paper-bypassed unless apply_in_paper_mode=True.
    Probes are point-sampled stdlib only (no psutil, no background task).
    rss_mb_max=0 disables the RSS check; loadavg_1m_max=0 disables loadavg.
    """
    enabled: bool = False
    apply_in_paper_mode: bool = False
    disk_pct_max: int = 90
    inode_pct_max: int = 90
    swap_pct_max: int = 50
    rss_mb_max: int = 0
    loadavg_1m_max: float = 0.0
    fail_closed_on_probe_error_in_capital_mode: bool = True


@dataclass(frozen=True, slots=True)
class ClockHealthCfg:
    """Phase 4.16 — clock-health admission gate.

    Disabled by default. Paper-bypassed unless apply_in_paper_mode=True.
    Detects monotonic/wall-clock skew, wall-clock regressions, and (in
    capital mode only, if require_ntp_sync_in_capital_mode=True) NTP
    synchronization via the `timedatectl` binary.
    """
    enabled: bool = False
    apply_in_paper_mode: bool = False
    require_ntp_sync_in_capital_mode: bool = True
    max_monotonic_wall_skew_ms: int = 2000
    reject_wall_clock_regression: bool = True
    timedatectl_timeout_sec: float = 2.0
    fail_closed_on_probe_error_in_capital_mode: bool = True


@dataclass(frozen=True, slots=True)
class TelegramWatchdogCfg:
    """Phase 4.14c — Telegram polling watchdog.

    Detects stalls in the TelegramBot.getUpdates loop. Monitors
    TelegramBot.last_activity_ts() every ``poll_interval_sec``. When the
    gap exceeds ``stall_threshold_sec``, restarts the bot task; if
    restarts within ``restart_window_sec`` exceed ``max_restarts`` and
    ``escalate_on_max`` is True, engages the kill-switch in SOFT mode.
    """
    enabled: bool = True
    stall_threshold_sec: int = 120
    poll_interval_sec: int = 10
    max_restarts: int = 3
    restart_window_sec: int = 300
    escalate_on_max: bool = True


@dataclass(frozen=True, slots=True)
class TelegramCfg:
    watchdog: TelegramWatchdogCfg = field(default_factory=TelegramWatchdogCfg)


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
    dead_man: DeadManCfg = field(default_factory=DeadManCfg)
    telegram: TelegramCfg = field(default_factory=TelegramCfg)
    fee_gate: FeeGateCfg = field(default_factory=FeeGateCfg)
    order_policy: OrderPolicyCfg = field(default_factory=OrderPolicyCfg)
    host_health: HostHealthCfg = field(default_factory=HostHealthCfg)
    clock_health: ClockHealthCfg = field(default_factory=ClockHealthCfg)

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
            dead_man=_parse_dead_man(raw.get("dead_man", {})),
            telegram=_parse_telegram(raw.get("telegram", {})),
            fee_gate=_parse_fee_gate(raw.get("fee_gate", {})),
            order_policy=_parse_order_policy(raw.get("order_policy", {})),
            host_health=_parse_host_health(raw.get("host_health", {})),
            clock_health=_parse_clock_health(raw.get("clock_health", {})),
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


def _parse_dead_man(d: dict[str, Any]) -> DeadManCfg:
    """Phase 4.14b: parse + bounds-check dead-man config.

    Hard limits: 1 <= min_timeout_sec <= default_timeout_sec <=
    max_timeout_sec <= 86400s (1d). The 86400 cap is a defensive
    upper bound; the spec recommends 43200s (12h) and below.
    """
    enabled = bool(d.get("enabled", False))
    min_t = _require_positive_int(
        d.get("min_timeout_sec", 300), "dead_man.min_timeout_sec", max_val=86400
    )
    default_t = _require_positive_int(
        d.get("default_timeout_sec", 21600),
        "dead_man.default_timeout_sec",
        max_val=86400,
    )
    max_t = _require_positive_int(
        d.get("max_timeout_sec", 43200), "dead_man.max_timeout_sec", max_val=86400
    )
    if not (min_t <= default_t <= max_t):
        raise ConfigError(
            f"dead_man bounds violated: require min_timeout_sec <= "
            f"default_timeout_sec <= max_timeout_sec, got "
            f"min={min_t} default={default_t} max={max_t}"
        )
    return DeadManCfg(
        enabled=enabled,
        default_timeout_sec=default_t,
        min_timeout_sec=min_t,
        max_timeout_sec=max_t,
    )


def _parse_telegram_watchdog(d: dict[str, Any]) -> TelegramWatchdogCfg:
    """Phase 4.14c: parse + bounds-check Telegram watchdog config.

    Bounds: stall_threshold_sec in [10, 3600]; poll_interval_sec in
    [1, 60] and < stall_threshold_sec; max_restarts in [0, 10];
    restart_window_sec in [60, 3600].
    """
    enabled = bool(d.get("enabled", True))
    stall_threshold_sec = _require_positive_int(
        d.get("stall_threshold_sec", 120),
        "telegram.watchdog.stall_threshold_sec",
        max_val=3600,
    )
    if stall_threshold_sec < 10:
        raise ConfigError(
            f"telegram.watchdog.stall_threshold_sec must be >= 10, got {stall_threshold_sec}"
        )
    poll_interval_sec = _require_positive_int(
        d.get("poll_interval_sec", 10),
        "telegram.watchdog.poll_interval_sec",
        max_val=60,
    )
    if poll_interval_sec >= stall_threshold_sec:
        raise ConfigError(
            f"telegram.watchdog.poll_interval_sec ({poll_interval_sec}) must be < "
            f"stall_threshold_sec ({stall_threshold_sec})"
        )
    max_restarts = _require_non_negative_int(
        d.get("max_restarts", 3),
        "telegram.watchdog.max_restarts",
        max_val=10,
    )
    restart_window_sec = _require_positive_int(
        d.get("restart_window_sec", 300),
        "telegram.watchdog.restart_window_sec",
        max_val=3600,
    )
    if restart_window_sec < 60:
        raise ConfigError(
            f"telegram.watchdog.restart_window_sec must be >= 60, got {restart_window_sec}"
        )
    escalate_on_max = bool(d.get("escalate_on_max", True))
    return TelegramWatchdogCfg(
        enabled=enabled,
        stall_threshold_sec=stall_threshold_sec,
        poll_interval_sec=poll_interval_sec,
        max_restarts=max_restarts,
        restart_window_sec=restart_window_sec,
        escalate_on_max=escalate_on_max,
    )


def _parse_telegram(d: dict[str, Any]) -> TelegramCfg:
    return TelegramCfg(watchdog=_parse_telegram_watchdog(d.get("watchdog", {})))


def _require_non_negative_float(
    value: Any, field_name: str, *, max_val: float | None = None
) -> float:
    """Coerce to float; require value >= 0; optional upper bound."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number, got {value!r}") from exc
    if v < 0:
        raise ConfigError(f"{field_name} must be non-negative, got {v}")
    if max_val is not None and v > max_val:
        raise ConfigError(f"{field_name} must be <= {max_val}, got {v}")
    return v


def _require_positive_float(
    value: Any, field_name: str, *, max_val: float | None = None
) -> float:
    """Coerce to float; require value > 0; optional upper bound."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number, got {value!r}") from exc
    if v <= 0:
        raise ConfigError(f"{field_name} must be positive, got {v}")
    if max_val is not None and v > max_val:
        raise ConfigError(f"{field_name} must be <= {max_val}, got {v}")
    return v


def _require_non_negative_decimal(value: Any, field_name: str) -> Decimal:
    """Coerce to Decimal; require value >= 0."""
    try:
        v = Decimal(str(value))
    except Exception as exc:
        raise ConfigError(f"{field_name} must be a number, got {value!r}") from exc
    if v < 0:
        raise ConfigError(f"{field_name} must be non-negative, got {v}")
    return v


def _parse_fee_gate(d: dict[str, Any]) -> FeeGateCfg:
    """Phase 4.15: parse fee_gate config.

    Enforces non-negative bps for default, safety margin, and every
    per-market / per-series override.
    """
    enabled = bool(d.get("enabled", True))
    apply_in_paper_mode = bool(d.get("apply_in_paper_mode", False))
    default_fee_bps = _require_non_negative_decimal(
        d.get("default_fee_bps", 0), "fee_gate.default_fee_bps"
    )
    safety_margin_bps = _require_non_negative_decimal(
        d.get("safety_margin_bps", 0), "fee_gate.safety_margin_bps"
    )
    per_market: dict[str, Decimal] = {}
    for k, v in (d.get("per_market_fee_bps") or {}).items():
        per_market[str(k)] = _require_non_negative_decimal(
            v, f"fee_gate.per_market_fee_bps[{k!r}]"
        )
    per_series: dict[str, Decimal] = {}
    for k, v in (d.get("per_series_fee_bps") or {}).items():
        per_series[str(k)] = _require_non_negative_decimal(
            v, f"fee_gate.per_series_fee_bps[{k!r}]"
        )
    return FeeGateCfg(
        enabled=enabled,
        apply_in_paper_mode=apply_in_paper_mode,
        default_fee_bps=default_fee_bps,
        safety_margin_bps=safety_margin_bps,
        per_market_fee_bps=per_market,
        per_series_fee_bps=per_series,
    )


def _parse_order_policy(d: dict[str, Any]) -> OrderPolicyCfg:
    """Phase 4.15: parse order_policy config.

    allowed_time_in_force is normalized to uppercase. An empty list is
    rejected — that would mean every intent is unconditionally rejected.
    """
    enabled = bool(d.get("enabled", True))
    apply_in_paper_mode = bool(d.get("apply_in_paper_mode", False))
    raw_tif = d.get("allowed_time_in_force", ("IOC", "FOK"))
    if not isinstance(raw_tif, (list, tuple)):
        raise ConfigError(
            f"order_policy.allowed_time_in_force must be a list, got {type(raw_tif).__name__}"
        )
    if len(raw_tif) == 0:
        raise ConfigError(
            "order_policy.allowed_time_in_force must be non-empty; "
            "empty list would reject every intent"
        )
    allowed = tuple(str(x).upper() for x in raw_tif)
    return OrderPolicyCfg(
        enabled=enabled,
        apply_in_paper_mode=apply_in_paper_mode,
        allowed_time_in_force=allowed,
        forbid_post_only=bool(d.get("forbid_post_only", True)),
        forbid_reduce_only=bool(d.get("forbid_reduce_only", False)),
        require_order_group_id_in_capital_mode=bool(
            d.get("require_order_group_id_in_capital_mode", True)
        ),
        require_buy_max_cost_for_buys_in_capital_mode=bool(
            d.get("require_buy_max_cost_for_buys_in_capital_mode", True)
        ),
    )


def _parse_host_health(d: dict[str, Any]) -> HostHealthCfg:
    """Phase 4.16: parse host_health config.

    Percent thresholds (disk/inode/swap) bounded to [0, 100]. rss_mb_max
    and loadavg_1m_max accept 0 to disable; otherwise non-negative.
    """
    return HostHealthCfg(
        enabled=bool(d.get("enabled", False)),
        apply_in_paper_mode=bool(d.get("apply_in_paper_mode", False)),
        disk_pct_max=_require_non_negative_int(
            d.get("disk_pct_max", 90), "host_health.disk_pct_max", max_val=100
        ),
        inode_pct_max=_require_non_negative_int(
            d.get("inode_pct_max", 90), "host_health.inode_pct_max", max_val=100
        ),
        swap_pct_max=_require_non_negative_int(
            d.get("swap_pct_max", 50), "host_health.swap_pct_max", max_val=100
        ),
        rss_mb_max=_require_non_negative_int(
            d.get("rss_mb_max", 0), "host_health.rss_mb_max", max_val=10_000_000
        ),
        loadavg_1m_max=_require_non_negative_float(
            d.get("loadavg_1m_max", 0.0),
            "host_health.loadavg_1m_max",
            max_val=10_000.0,
        ),
        fail_closed_on_probe_error_in_capital_mode=bool(
            d.get("fail_closed_on_probe_error_in_capital_mode", True)
        ),
    )


def _parse_clock_health(d: dict[str, Any]) -> ClockHealthCfg:
    """Phase 4.16: parse clock_health config.

    max_monotonic_wall_skew_ms must be > 0; the gate rejects when an
    observed skew exceeds this. timedatectl_timeout_sec must be > 0
    (subprocess timeout).
    """
    return ClockHealthCfg(
        enabled=bool(d.get("enabled", False)),
        apply_in_paper_mode=bool(d.get("apply_in_paper_mode", False)),
        require_ntp_sync_in_capital_mode=bool(
            d.get("require_ntp_sync_in_capital_mode", True)
        ),
        max_monotonic_wall_skew_ms=_require_positive_int(
            d.get("max_monotonic_wall_skew_ms", 2000),
            "clock_health.max_monotonic_wall_skew_ms",
            max_val=86_400_000,
        ),
        reject_wall_clock_regression=bool(
            d.get("reject_wall_clock_regression", True)
        ),
        timedatectl_timeout_sec=_require_positive_float(
            d.get("timedatectl_timeout_sec", 2.0),
            "clock_health.timedatectl_timeout_sec",
            max_val=60.0,
        ),
        fail_closed_on_probe_error_in_capital_mode=bool(
            d.get("fail_closed_on_probe_error_in_capital_mode", True)
        ),
    )


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
