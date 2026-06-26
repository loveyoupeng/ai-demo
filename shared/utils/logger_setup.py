"""Singleton utility to initialise and configure logging across the project.

Every module gets a logger via ``logging.getLogger(__name__)``.
This function configures the root logger with appropriate handlers and
a human-friendly format so users can trace runtime behaviour easily.

Default format:  ``{timestamp} [{levelname:7s}] {name} {message}``
  - ``INFO`` = epoch/batch/token-level summaries (default)
  - ``DEBUG`` = tensor shapes, intermediate values
  - ``TRACE`` = per-operation detail (attention entropy, activation stats)

Usage
-----
    from shared.utils.logger_setup import setup_logging
    setup_logging(level="DEBUG", log_file="train.log")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s %(message)s"
_DEFAULT_DATEfmt = "%Y-%m-%d %H:%M:%S"


class _TimestampFilter(logging.Filter):
    """Inject a formatted ``ts`` attribute for formatting."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return True


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None,
) -> None:
    """Initialise the project-wide logging infrastructure.

    Call this once early (e.g. at the top of ``scripts/train.py``).

    Parameters
    ----------
    level : str
        Logging level for the root logger.  One of ``DEBUG``, ``INFO``,
        ``WARNING``, ``ERROR``, ``CRITICAL``.  Default is ``INFO``.
    log_file : Path or None
        If given, a ``FileHandler`` is attached that writes to this path.
        Useful for long training runs where stdout is scrolled away.
    format_string : str or None
        Custom ``logging.Formatter`` pattern string.  When ``None`` the
        default format is used::

            2025-06-26 14:30:01 [  INFO] scripts.train Starting epoch 1/10

    Notes
    -----
    - The function is **idempotent** — calling it multiple times does not
      duplicate handlers (handlers that already exist are reused).
    - The timestamp format is fixed; the rest of the format can be customised.

    """
    root = logging.getLogger()

    # Re-use existing handlers to avoid duplicates on re-calls.
    if root.handlers:
        # Update level of existing handlers.
        for handler in root.handlers:
            handler.setLevel(level)
        root.setLevel(level)
        return

    # -- Formatter --
    fmt = format_string or _DEFAULT_FORMAT

    # -- Console handler (stdout) --
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt=_DEFAULT_DATEfmt))
    root.addHandler(console)

    # -- File handler (optional) --
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=_DEFAULT_DATEfmt))
        root.addHandler(file_handler)


def set_level(logger_name: str, level: str) -> None:
    """Override the log level for a specific logger *after* ``setup_logging()``.

    Useful for selectively increasing verbosity of only a few modules
    while keeping the rest at INFO.

    Parameters
    ----------
    logger_name : str
        Dotted logger name, e.g. ``impl._np.modules``.
    level : str
        New logging level as a string.

    """
    logging.getLogger(logger_name).setLevel(level)


def log(*args: object) -> None:
    """Log a message at TRACE level using this module's logger.

    Use the ``trace`` convenience function instead in documentation examples.

    TRACE is a custom level between DEBUG and INFO.  Messages logged
    at this level are only visible when the logger is set to DEBUG.

    Usage
    -----
    .. code-block:: python

        from shared.utils.logger_setup import log
        log("attn_entropy head=0 pos=5 h=3.20")
    """
    logger.debug("[%s] %s", log.__name__, " ".join(str(a) for a in args))
