"""Tests for the Langfuse observability layer."""

import os
from typing import Any

import pytest

from lib.sql_generator import observability
from lib.sql_generator.config import Settings, get_settings
from lib.sql_generator.observability import (
    _ensure_env,
    observe_run,
    observe_step,
    tracing_configured,
)


def _add(a: int, b: int) -> int:
    return a + b


def _enable_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.local")
    get_settings.cache_clear()


def test_tracing_configured_disabled_by_default() -> None:
    get_settings.cache_clear()
    try:
        assert tracing_configured() is False
    finally:
        get_settings.cache_clear()


def test_tracing_configured_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_langfuse_env(monkeypatch)
    try:
        assert tracing_configured() is True
    finally:
        get_settings.cache_clear()


def test_observe_step_noop_when_disabled() -> None:
    decorated = observe_step("understand_intent")(_add)
    assert decorated(2, 3) == 5


def test_observe_run_noop_when_disabled() -> None:
    decorated = observe_run("sql_generator_agent")(_add)
    assert decorated(1, 1) == 2


@pytest.mark.parametrize(
    ("factory", "expected_type"),
    [(observe_step, "span"), (observe_run, "chain")],
)
def test_observe_decorators_enable_tracing(
    monkeypatch: pytest.MonkeyPatch,
    factory: Any,
    expected_type: str,
) -> None:
    recorded: list[tuple[str, str]] = []

    def fake_observe(*, name: str | None = None, as_type: str | None = None) -> Any:
        recorded.append((name or "", as_type or ""))
        return lambda func: func

    monkeypatch.setattr("lib.sql_generator.observability.langfuse_observe", fake_observe)
    _enable_langfuse_env(monkeypatch)
    decorated = factory("my_step")(_add)
    assert decorated(3, 4) == 7
    assert recorded == [("my_step", expected_type)]
    assert os.environ["LANGFUSE_ENABLED"] == "true"


def test_ensure_env_sets_missing_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(observability, "load_dotenv", lambda: None)
    settings = Settings(
        langfuse_enabled=True,
        langfuse_public_key="pk-settings",
        langfuse_secret_key="sk-settings",
        langfuse_host="https://settings.local",
    )
    _ensure_env(settings)
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-settings"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-settings"
    assert os.environ["LANGFUSE_HOST"] == "https://settings.local"
    assert os.environ["LANGFUSE_ENABLED"] == "true"


def test_ensure_env_keeps_existing_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(observability, "load_dotenv", lambda: None)
    _enable_langfuse_env(monkeypatch)
    settings = Settings(
        langfuse_enabled=False,
        langfuse_public_key="pk-other",
        langfuse_secret_key="sk-other",
        langfuse_host="https://other.local",
    )
    _ensure_env(settings)
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-test"
    assert os.environ["LANGFUSE_ENABLED"] == "false"
