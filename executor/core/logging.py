"""
Structured logging setup.

Spec: "Structured logging on every await point." structlog is configured for JSON
output on stdout (systemd/journalctl captures it) with ts, level, logger name,
and any bound context (strategy_id, intent_id, venue, etc.).
"""
from __future__ import annotations

import logging
import os
import sys

import structlog


_CONFIGURED = False


def configure(level: str | None = None, json_output: bool | None = None) -> None:
    """Configure structlog + stdlib logging. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    lvl_name = (level or os.environ.get("EXECUTOR_LOG_LEVEL", "INFO")).upper()
    lvl = getattr(logging, lvl_name, logging.INFO)

    # Default: JSON when not a tty (systemd/journalctl), pretty when tty.
    if json_output is None:
        json_output = not sys.stdout.isatty()

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Quiet stdlib root — we route through structlog where we care.
    logging.basicConfig(level=lvl, stream=sys.stdout, format="%(message)s")

    _CONFIGURED = True


def get_logger(name: str):
    if not _CONFIGURED:
        configure()
    # Bind the logger name as a context field — structlog's PrintLogger has
    # no .name attribute so we can't use add_logger_name.
    return structlog.get_logger().bind(logger=name)
