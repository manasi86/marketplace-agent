"""OpenAI-compatible LLM client factory shared by every agent."""

import logging

from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agents.common.config import Settings

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
    """Invoke the model with a prompt and return the trimmed text response."""
    model_name = getattr(llm, "model_name", None) or llm.__class__.__name__
    logger.info("LLM invoke (%s, %d chars in prompt)", model_name, len(prompt))
    logger.debug("Prompt:\n%s", prompt)
    response = llm.invoke(prompt)
    if not isinstance(response, str):
        # LangChain message objects expose their content via .content.
        content = getattr(response, "content", None)
        if content is None:
            raise LLMError("Model returned an empty response.")
        response = str(content).strip()
    else:
        response = response.strip()
    logger.debug("Response:\n%s", response)
    return response
