"""OpenAI-compatible LLM client factory shared by every agent."""

import logging
from typing import Any

from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agents.common.config import Settings
from agents.common.observability import observe_generation

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM cannot be configured or invoked."""


def get_llm(settings: Settings) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at the configured endpoint."""
    if not settings.has_llm_credentials:
        raise LLMError("Missing LLM API key. Set SQL_GEN_API_KEY in the .env file.")
    logger.info(
        "Initializing LLM client (model=%s, base_url=%s)",
        settings.sql_gen_model,
        settings.sql_gen_base_url,
    )
    return ChatOpenAI(
        model=settings.sql_gen_model,
        api_key=SecretStr(settings.sql_gen_api_key),
        base_url=settings.sql_gen_base_url,
        temperature=0,
        timeout=60,
        max_retries=2,
    )


def invoke_llm(llm: LanguageModelLike, prompt: str) -> str:
    """Invoke the model with a prompt and return the trimmed text response.

    The call is recorded in Langfuse as a generation carrying the model name
    and input/output token usage when observability is enabled.
    """
    model_name = getattr(llm, "model_name", None) or llm.__class__.__name__
    logger.info("LLM invoke (%s, %d chars in prompt)", model_name, len(prompt))
    logger.debug("Prompt:\n%s", prompt)
    with observe_generation("llm_call", prompt) as generation:
        response = llm.invoke(prompt)
        text = _content_text(response)
        generation.update(
            output=text,
            model=_response_model(response, fallback=model_name),
            usage_details=_usage_details(response),
        )
    logger.debug("Response:\n%s", text)
    return text


def _content_text(response: Any) -> str:
    """Extract and trim the textual content of an LLM response."""
    if isinstance(response, str):
        text = response.strip()
    else:
        # LangChain message objects expose their content via .content.
        content = getattr(response, "content", None)
        if content is None:
            raise LLMError("Model returned an empty response.")
        text = str(content).strip()
    return text


def _response_model(response: Any, *, fallback: str) -> str:
    """Return the model that produced the response, falling back when unknown."""
    metadata = getattr(response, "response_metadata", None) or {}
    if isinstance(metadata, dict):
        return str(metadata.get("model_name") or metadata.get("model") or fallback)
    return fallback


def _usage_details(response: Any) -> dict[str, int] | None:
    """Map LangChain usage_metadata onto Langfuse usage_details keys."""
    usage = getattr(response, "usage_metadata", None) or {}
    if not isinstance(usage, dict) or not usage:
        return None
    details: dict[str, int] = {}
    prompt_tokens = _int_value(usage, "input_tokens") or _int_value(usage, "prompt_tokens")
    output_tokens = _int_value(usage, "output_tokens") or _int_value(usage, "completion_tokens")
    total_tokens = _int_value(usage, "total_tokens")
    if prompt_tokens is not None:
        details["prompt_tokens"] = prompt_tokens
    if output_tokens is not None:
        details["completion_tokens"] = output_tokens
    if total_tokens is not None:
        details["total_tokens"] = total_tokens
    return details or None


def _int_value(mapping: Any, key: str) -> int | None:
    value = mapping.get(key)
    return int(value) if isinstance(value, (int, float)) else None
