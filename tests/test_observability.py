"""Tests for the Langfuse observability layer."""

import os
from typing import Any

import pytest

from agents.common import observability
from agents.common.config import Settings, get_settings
from agents.common.observability import (
    _ensure_env,
    observe_generation,
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

    monkeypatch.setattr("agents.common.observability.langfuse_observe", fake_observe)
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


def test_observe_generation_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_client() -> Any:
        raise AssertionError("langfuse client must not be created when disabled")

    monkeypatch.setattr(observability, "langfuse_get_client", unexpected_client)
    get_settings.cache_clear()
    try:
        with observe_generation("llm_call", "a prompt") as generation:
            generation.update(model="gpt-4o", usage_details={"prompt_tokens": 1})
    finally:
        get_settings.cache_clear()


def test_observe_generation_enabled_uses_generation_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[tuple[str, str, str]] = []

    class _FakeGeneration:
        def update(self, **_: Any) -> None:
            return None

        def __enter__(self) -> "_FakeGeneration":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    class _Fake:
        def start_as_current_observation(self, **kwargs: Any) -> _FakeGeneration:
            recorded.append(
                (
                    str(kwargs.get("name")),
                    str(kwargs.get("as_type")),
                    str(kwargs.get("input")),
                )
            )
            return _FakeGeneration()

    def fake_client() -> Any:
        return _Fake()

    monkeypatch.setattr(observability, "langfuse_get_client", fake_client)
    _enable_langfuse_env(monkeypatch)
    try:
        with observe_generation("llm_call", "classify me") as generation:
            generation.update(model="gpt-4o", usage_details={"prompt_tokens": 3})
        assert recorded == [("llm_call", "generation", "classify me")]
    finally:
        get_settings.cache_clear()
