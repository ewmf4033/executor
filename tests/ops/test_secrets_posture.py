"""Phase 4.17 secrets posture guardrails.

On the production droplet, verifies:
- ``/root/executor/.env`` exists, has no group/world bits, and is root-owned.
- No ``.env.bak.*`` leftovers in the repo working dir.
- ``/root/kalshi_sports.key`` and ``/root/kalshi_private.key`` (if present)
  have no group/world bits and are root-owned.
- No obvious secret-shaped strings appear in tracked source directories.

The whole module skips on hosts without ``/root/executor`` (CI / dev laptops).
The source-scan reports file path + line number only — never the matched
value — so a failure cannot itself leak the secret it found.

Patterns are deliberately conservative (long, distinctive shapes) to keep
false positives near zero. Add new patterns here when a new secret class
enters the system; do NOT relax existing patterns to "fix" a hit — rotate
the leaked secret and remove it from the source instead.
"""
from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest


HOST_GUARD = Path("/root/executor")
ENV_FILE = HOST_GUARD / ".env"
ENV_BAK_GLOB = ".env.bak.*"
KALSHI_KEY_FILES = (
    Path("/root/kalshi_sports.key"),
    Path("/root/kalshi_private.key"),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("executor", "scripts", "systemd", "config", "tests")
EXCLUDE_DIR_PARTS = {".venv", ".git", "__pycache__", "audit-logs", "reviews"}
# Files we will read as text. Anything else is skipped (binary safety).
TEXT_EXTS = {
    ".py", ".sh", ".service", ".env", ".yaml", ".yml", ".toml", ".cfg",
    ".ini", ".md", ".txt", ".conf", ".example", ".j2", "",
}

# Names appear in error output; matched values NEVER do.
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "telegram_bot_token": re.compile(r"\b\d{8,}:[A-Za-z0-9_-]{30,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "aws_access_key_id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "pem_private_key": re.compile(
        r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"
    ),
}

pytestmark = pytest.mark.skipif(
    not HOST_GUARD.exists(),
    reason="not on production host (no /root/executor)",
)


def test_env_file_mode_no_group_or_world() -> None:
    """`.env` must not be readable by group or world, and must be root-owned."""
    if not ENV_FILE.exists():
        pytest.skip(f"{ENV_FILE} not present")
    st = ENV_FILE.stat()
    perms = stat.S_IMODE(st.st_mode)
    assert perms & 0o077 == 0, (
        f"{ENV_FILE} mode {oct(perms)} exposes group/world bits "
        f"— must be 0400 (or 0600) with no group/world access"
    )
    assert st.st_uid == 0, f"{ENV_FILE} not root-owned (uid={st.st_uid})"


def test_no_env_bak_left_in_repo() -> None:
    """Stale `.env.bak.*` backups must live outside the repo (e.g. /root/secret-archives)."""
    leftovers = sorted(HOST_GUARD.glob(ENV_BAK_GLOB))
    paths = [str(p) for p in leftovers]
    assert not leftovers, (
        f"stale .env backup files in repo: {paths} — "
        "move to /root/secret-archives (or delete) and rerun"
    )


@pytest.mark.parametrize("key_path", KALSHI_KEY_FILES)
def test_kalshi_private_key_mode(key_path: Path) -> None:
    """Kalshi private key files (if present) must be root-only readable."""
    if not key_path.exists():
        pytest.skip(f"{key_path} not present")
    st = key_path.stat()
    perms = stat.S_IMODE(st.st_mode)
    assert perms & 0o077 == 0, (
        f"{key_path} mode {oct(perms)} exposes group/world bits "
        f"— must be 0600 or stricter"
    )
    assert st.st_uid == 0, f"{key_path} not root-owned (uid={st.st_uid})"


def _iter_scan_files() -> "list[Path]":
    out: list[Path] = []
    for top in SCAN_DIRS:
        base = REPO_ROOT / top
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if any(part in EXCLUDE_DIR_PARTS for part in path.parts):
                continue
            if path.suffix.lower() not in TEXT_EXTS:
                continue
            out.append(path)
    return out


def test_no_obvious_secret_strings_in_source() -> None:
    """Scan tracked source dirs for secret-shaped strings.

    Reports only ``path:line [pattern_name]`` on failure — never the value.
    """
    self_path = Path(__file__).resolve()
    hits: list[str] = []
    for path in _iter_scan_files():
        if path.resolve() == self_path:
            # Don't scan the scanner — the regex literals here are constructed
            # to never match themselves, but skipping documents intent.
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, rx in SECRET_PATTERNS.items():
                if rx.search(line):
                    rel = path.relative_to(REPO_ROOT)
                    hits.append(f"{rel}:{lineno} [{name}]")
                    break  # one hit per line is enough; don't double-report
    assert not hits, (
        "secret-shaped strings found in tracked source (values NOT shown):\n  "
        + "\n  ".join(hits)
        + "\n\nIf the hit is a real secret: rotate it, remove from source, "
        "and re-run. If it is a synthetic test fixture, make it less "
        "secret-shaped (e.g. shorter, no `:` separator)."
    )
