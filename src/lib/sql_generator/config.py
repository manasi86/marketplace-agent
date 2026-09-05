"""Environment-based configuration for the SQL generator agent."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

MODULE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(
        env_file=(Path.cwd() / ".env", MODULE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM.
    sql_gen_base_url: str = "https://llm.nalits.com/v1"
    sql_gen_model: str = "gpt-4o"
    sql_gen_api_key: str = ""

    # Oracle database.
    oracle_dsn: str = ""
    oracle_user: str = ""
    oracle_password: str = ""

    # Langfuse (self-hosted).
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_enabled: bool = True

    # Agent behaviour.
    max_sql_attempts: int = 3

    @property
    def has_oracle_credentials(self) -> bool:
        """Return True when all Oracle connection settings are provided."""
        return bool(self.oracle_dsn and self.oracle_user and self.oracle_password)

    @property
    def has_llm_credentials(self) -> bool:
        """Return True when an API key has been provided for the LLM."""
        return bool(self.sql_gen_api_key)

    @property
    def has_langfuse_credentials(self) -> bool:
        """Return True when self-hosted Langfuse credentials are present."""
        return bool(self.langfuse_host and self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
