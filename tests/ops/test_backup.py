"""Smoke test for the disaster-recovery backup script."""

import os
import subprocess
import tarfile

import pytest

BACKUP_SCRIPT = "/root/trading-wiki/scripts/backup_droplet.sh"


@pytest.mark.skipif(
    not os.path.isfile(BACKUP_SCRIPT),
    reason="backup script not found",
)
def test_backup_dry_run():
    """Run backup_droplet.sh --dry-run and verify the tarball contents.

    Acceptable outcomes:
      - exit 0: tar created successfully (dry-run skips upload)
      - exit 1: rclone not configured — acceptable in CI / fresh environments
    """
    result = subprocess.run(
        [BACKUP_SCRIPT, "--dry-run"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Exit 1 means rclone not configured — acceptable, nothing more to check
    if result.returncode == 1:
        assert "rclone" in result.stdout.lower() or "rclone" in result.stderr.lower(), (
            f"Exit 1 but no rclone message. stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        return

    assert result.returncode == 0, (
        f"Unexpected exit code {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    # Find the tarball that was created
    tarball_path = None
    for line in result.stdout.splitlines():
        if "/tmp/droplet-backup-" in line and ".tar.gz" in line:
            # Extract path from log line
            for token in line.split():
                if token.startswith("/tmp/droplet-backup-") and token.endswith(".tar.gz"):
                    tarball_path = token
                    break
            if tarball_path:
                break

    assert tarball_path is not None, (
        f"Could not find tarball path in output: {result.stdout!r}"
    )
    assert os.path.isfile(tarball_path), f"Tarball not found at {tarball_path}"

    # Verify tarball contents include expected components
    with tarfile.open(tarball_path, "r:gz") as tf:
        members = tf.getnames()

    expected_prefixes = [
        "executor/audit-logs/",
        "executor/state/",
        "trading-wiki/",
    ]
    for prefix in expected_prefixes:
        matches = [m for m in members if m.startswith(prefix)]
        assert matches, f"No entries found with prefix '{prefix}' in tarball"

    # Cleanup
    os.remove(tarball_path)
