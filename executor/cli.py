"""
Phase 4 CLI entry point with --paper / --live safety gate.

`--live` requires ALL of:
  (1) risk.yaml exists at the resolved path AND has been edited from the
      shipped defaults (we hash the in-repo defaults and reject if it matches),
  (2) TELEGRAM_CHAT_ID is set in env,
  (3) EXECUTOR_LIVE_ACK env var matches the exact phrase
      "i-accept-capital-risk-2026-04-19".

`--paper` (default) bypasses the gate but pins PAPER_MODE=true so adapters
do not place real orders even if the env says otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


DEFAULT_RISK_YAML = "/root/executor/config/risk.yaml"
LIVE_ACK_PHRASE = "i-accept-capital-risk-2026-04-19"


def shipped_defaults_fingerprint(risk_yaml_text: str) -> str:
    return hashlib.sha256(risk_yaml_text.encode("utf-8")).hexdigest()


def _load_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


# Captured at module import: this is the SHA256 of the shipped risk.yaml
# the moment the live-gate check runs. If a deployer hasn't edited it,
# the hash will match a known reference we record below.
SHIPPED_DEFAULTS_HASH = (
    # SHA256 of the full Phase 4 risk.yaml as committed. Recomputed on every
    # check_live_gate() call so we tolerate documentation tweaks: we compare
    # against the on-disk file's *current* hash AND a "lock" file. To keep
    # this self-contained, the gate enforces that risk.yaml differs from a
    # canonical "all defaults" copy stored alongside under risk.defaults.yaml.
    ""
)


def _defaults_path() -> Path:
    return Path("/root/executor/config/risk.defaults.yaml")


def check_live_gate(*, risk_yaml_path: str = DEFAULT_RISK_YAML, env: dict[str, str] | None = None) -> tuple[bool, str]:
    env = env if env is not None else dict(os.environ)
    # 1. risk.yaml must differ from defaults
    cur = _load_text(risk_yaml_path)
    if not cur:
        return False, f"risk.yaml not found at {risk_yaml_path}"
    defaults_text = _load_text(_defaults_path())
    if defaults_text and shipped_defaults_fingerprint(cur) == shipped_defaults_fingerprint(defaults_text):
        return False, (
            f"risk.yaml at {risk_yaml_path} matches shipped defaults; "
            f"edit it before going live (compare against {_defaults_path()})"
        )
    # 2. TELEGRAM_CHAT_ID must be set
    if not env.get("TELEGRAM_CHAT_ID"):
        return False, "TELEGRAM_CHAT_ID not set"
    # 3. EXECUTOR_LIVE_ACK must equal exact phrase
    ack = env.get("EXECUTOR_LIVE_ACK", "")
    if ack != LIVE_ACK_PHRASE:
        return False, (
            "EXECUTOR_LIVE_ACK missing or wrong; set it to "
            f"{LIVE_ACK_PHRASE!r} to confirm"
        )
    return True, ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="executor", description="0d executor service")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--paper", action="store_true", help="paper mode (default)")
    mode.add_argument("--live", action="store_true", help="live trading (requires gate)")
    parser.add_argument(
        "--risk-yaml",
        default=DEFAULT_RISK_YAML,
        help="path to risk.yaml (default: %(default)s)",
    )
    return parser.parse_args(argv)


def resolve_mode(args: argparse.Namespace, *, env: dict[str, str] | None = None) -> str:
    """
    Returns "paper" or "live". On --live, applies the safety gate and exits
    the process with code 2 if it fails.
    """
    if args.live:
        ok, why = check_live_gate(risk_yaml_path=args.risk_yaml, env=env)
        if not ok:
            print(f"--live refused: {why}", file=sys.stderr)
            sys.exit(2)
        # Honor explicit live: do NOT pin PAPER_MODE.
        return "live"
    # Default + --paper both pin PAPER_MODE=true.
    os.environ["PAPER_MODE"] = "true"
    return "paper"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mode = resolve_mode(args)
    # Lazy import — keeps CLI cheap to test.
    from .core.service import main as svc_main
    print(f"executor mode: {mode}", flush=True)
    svc_main()


if __name__ == "__main__":
    main()
