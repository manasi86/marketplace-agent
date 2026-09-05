"""Console logging configuration for the SQL generator agent."""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "langchain",
    "langgraph",
    "langfuse",
    "oracledb",
)


class _ConsoleHandler(logging.Handler):
    """Write every record to the *current* ``sys.stderr``.

    Resolving ``sys.stderr`` at emit time (rather than binding it at creation)
    keeps logging aligned with stream redirection, e.g. pytest's ``capsys``.
    """

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[explicit-override]
        try:
            sys.stderr.write(f"{self.format(record)}\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)


def configure_logging(*, verbose: bool = False) -> None:
    """Install a detailed console logger and set the root log level.

    ``verbose=False`` selects INFO, ``verbose=True`` selects DEBUG (which also
    surfaces full prompts, SQL and model responses). Quiet third-party loggers
    are bumped to WARNING so the terminal stays readable.
    """
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(handler, _ConsoleHandler) for handler in root.handlers):
        console = _ConsoleHandler()
        console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(console)
    for handler in root.handlers:
        handler.setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
