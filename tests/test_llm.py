"""Tests for the LLM client factory and invocation helper."""

from langchain_openai import ChatOpenAI
import pytest

from lib.sql_generator.config import Settings
from lib.sql_generator.llm import LLMError, get_llm, invoke_llm


class _MessageLike:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _NoContent:
    pass


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
