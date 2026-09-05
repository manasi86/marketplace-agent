"""Tests for the console logging configuration."""

import logging

import pytest

from agents.common.logging_setup import _NOISY_LOGGERS, _ConsoleHandler, configure_logging


def _console_handlers() -> list[_ConsoleHandler]:
    return [h for h in logging.getLogger().handlers if isinstance(h, _ConsoleHandler)]


def test_configure_logging_adds_console_handler() -> None:
    configure_logging()
    console = _console_handlers()
    assert len(console) == 1
    assert console[0].level == logging.INFO


def test_configure_logging_verbose_selects_debug_level() -> None:
    configure_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG
    assert all(handler.level == logging.DEBUG for handler in _console_handlers())


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    configure_logging()
    assert len(_console_handlers()) == 1


def test_configure_logging_quiets_noisy_loggers() -> None:
    configure_logging()
    for name in _NOISY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


def test_console_handler_writes_to_current_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    logging.getLogger("test").info("visible log line")
    assert "visible log line" in capsys.readouterr().err


def test_console_handler_swallows_emit_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _ConsoleHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    class BrokenStream:
        def write(self, text: str) -> None:
            del text
            raise OSError("stream is closed")

        def flush(self) -> None:
            raise OSError("stream is closed")

    monkeypatch.setattr("sys.stderr", BrokenStream())
    monkeypatch.setattr(logging, "raiseExceptions", False)
    record = logging.LogRecord("test", logging.ERROR, __name__, 0, "boom", None, None)
    handler.emit(record)
