"""
executor.control — operator control plane.

Out-of-process control channel for the daemon. Phase 4.14a provides an
AF_UNIX socket listener inside the daemon + a standalone stdlib CLI
(scripts/executorctl) so the operator can engage/resume kill even when
Telegram is unreachable. Socket auth is filesystem permissions
(User=root + mode 0600); remote access is via SSH/Tailscale tunnel —
no public HTTPS endpoint per GPT-5.5 review #2.
"""
