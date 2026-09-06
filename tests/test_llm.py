"""Tests for the LLM client factory and invocation helper."""

from typing import Any

from langchain_openai import ChatOpenAI
import pytest

from agents.common import observability
from agents.common.config import Settings, get_settings
from agents.common.llm import LLMError, get_llm, invoke_llm


class _MessageLike:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _NoContent:
    pass


class _RecordingGeneration:
    """Records update() kwargs and absorbs the context manager protocol."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def __enter__(self) -> "_RecordingGeneration":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _enable_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.local")
    get_settings.cache_clear()


def _settings(**overrides: str | int | bool) -> Settings:
    defaults: dict[str, str | int | bool] = {
        "sql_gen_api_key": "test-key",
        "sql_gen_base_url": "https://llm.nalits.com/v1",
        "sql_gen_model": "gpt-4o",
    }
    defaults.update(overrides)
    return Settings.model_validate(defaults)


def test_get_llm_requires_api_key() -> None:
    with pytest.raises(LLMError, match="SQL_GEN_API_KEY"):
        get_llm(_settings(sql_gen_api_key=""))


def test_get_llm_configures_client() -> None:
    client = get_llm(_settings())
    assert isinstance(client, ChatOpenAI)
    assert client.model_name == "gpt-4o"
    assert client.openai_api_base == "https://llm.nalits.com/v1"


def test_get_llm_creates_equivalent_clients() -> None:
    first = get_llm(_settings())
    second = get_llm(_settings())
    assert first is not second
    assert first.model_name == second.model_name


def test_invoke_llm_with_string_response() -> None:
    class _StringLLM:
        def invoke(self, prompt: str) -> str:
            del prompt
            return "  SELECT 1 FROM dual  "

    result = invoke_llm(_StringLLM(), "select one")  # type: ignore[arg-type]
    assert result == "SELECT 1 FROM dual"


def test_invoke_llm_with_message_like_response() -> None:
    class _MessageLLM:
        def invoke(self, prompt: str) -> _MessageLike:
            del prompt
            return _MessageLike("  42  ")

    result = invoke_llm(_MessageLLM(), "q")  # type: ignore[arg-type]
    assert result == "42"


def test_invoke_llm_empty_content_raises() -> None:
    class _EmptyLLM:
        def invoke(self, prompt: str) -> _MessageLike:
            del prompt
            return _MessageLike(None)

    with pytest.raises(LLMError, match="empty response"):
        invoke_llm(_EmptyLLM(), "q")  # type: ignore[arg-type]


def test_invoke_llm_no_content_attribute_raises() -> None:
    class _NoContentLLM:
        def invoke(self, prompt: str) -> _NoContent:
            del prompt
            return _NoContent()

    with pytest.raises(LLMError, match="empty response"):
        invoke_llm(_NoContentLLM(), "q")  # type: ignore[arg-type]


def test_invoke_llm_captures_model_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UsageMessage:
        content = "  Total sales were 100.  "
        usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        response_metadata = {"model_name": "gpt-4o-2026"}

    class _UsageLLM:
        def invoke(self, prompt: str) -> _UsageMessage:
            del prompt
            return _UsageMessage()

    _enable_langfuse_env(monkeypatch)
    generation = _RecordingGeneration()

    def fake_client() -> Any:
        class _Fake:
            def start_as_current_observation(self, **kwargs: Any) -> Any:
                assert kwargs["as_type"] == "generation"
                assert kwargs["input"] == "q"
                return generation

        return _Fake()

    monkeypatch.setattr(observability, "langfuse_get_client", fake_client)
    try:
        result = invoke_llm(_UsageLLM(), "q")  # type: ignore[arg-type]
        assert result == "Total sales were 100."
        assert generation.updates == [
            {
                "output": "Total sales were 100.",
                "model": "gpt-4o-2026",
                "usage_details": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        ]
    finally:
        get_settings.cache_clear()


def test_invoke_llm_skips_client_when_tracing_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StringLLM:
        def invoke(self, prompt: str) -> str:
            del prompt
            return " 42  "

    def unexpected_client() -> Any:
        raise AssertionError("langfuse client must not be created when disabled")

    monkeypatch.setattr(observability, "langfuse_get_client", unexpected_client)
    get_settings.cache_clear()
    try:
        result = invoke_llm(_StringLLM(), "q")  # type: ignore[arg-type]
        assert result == "42"
    finally:
        get_settings.cache_clear()


def test_invoke_llm_falls_back_to_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PlainLLM:
        model_name = "configured-model"

        def invoke(self, prompt: str) -> str:
            del prompt
            return "12"

    _enable_langfuse_env(monkeypatch)
    generation = _RecordingGeneration()

    def fake_client() -> Any:
        class _Fake:
            def start_as_current_observation(self, **kwargs: Any) -> Any:
                return generation

        return _Fake()

    monkeypatch.setattr(observability, "langfuse_get_client", fake_client)
    try:
        invoke_llm(_PlainLLM(), "q")  # type: ignore[arg-type]
        assert generation.updates[0]["model"] == "configured-model"
        assert generation.updates[0]["usage_details"] is None
    finally:
        get_settings.cache_clear()


def test_response_model_falls_back_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agents.common.llm import _response_model

    class _NonDictMetadata:
        response_metadata = "unexpected"

    class _NoMetadata:
        pass

    assert _response_model(_NonDictMetadata(), fallback="fb") == "fb"
    assert _response_model(_NoMetadata(), fallback="fb") == "fb"


def test_usage_details_partial_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.common.llm import _usage_details

    class _TotalOnly:
        usage_metadata: Any = {"total_tokens": 15}
        content = "x"
        response_metadata: Any = None

    details = _usage_details(_TotalOnly())
    assert details == {"total_tokens": 15}

    class _Invalid:
        usage_metadata: Any = {"total_tokens": "not-a-number"}
        content = "x"
        response_metadata: Any = None

    assert _usage_details(_Invalid()) is None

    class _Empty:
        content = "x"
        response_metadata: Any = None

    assert _usage_details(_Empty()) is None

    class _NonDict:
        usage_metadata: Any = "unexpected"
        content = "x"
        response_metadata: Any = None

    assert _usage_details(_NonDict()) is None
