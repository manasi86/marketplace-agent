"""OpenAI-compatible LLM client factory."""

from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from lib.sql_generator.config import Settings


class LLMError(Exception):
    """Raised when the LLM cannot be configured or invoked."""


def get_llm(settings: Settings) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at the configured endpoint."""
    if not settings.has_llm_credentials:
        raise LLMError("Missing LLM API key. Set SQL_GEN_API_KEY in the .env file.")
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
    response = llm.invoke(prompt)
    if not isinstance(response, str):
        # LangChain message objects expose their content via .content.
        content = getattr(response, "content", None)
        if content is None:
            raise LLMError("Model returned an empty response.")
        return str(content).strip()
    return response.strip()
