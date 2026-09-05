"""Tests for configuration loading."""

from pydantic_settings import BaseSettings
import pytest

from agents.common.config import Settings, get_settings


class TestSettings:
    """Unit tests for the Settings model."""

    def test_defaults(self) -> None:
        settings = Settings(
            sql_gen_base_url="https://llm.nalits.com/v1",
            sql_gen_model="gpt-4o",
            sql_gen_api_key="",
            oracle_dsn="",
            oracle_user="",
            oracle_password="",
            langfuse_host="",
            langfuse_public_key="",
            langfuse_secret_key="",
            langfuse_enabled=False,
            max_sql_attempts=3,
        )
        assert settings.sql_gen_base_url == "https://llm.nalits.com/v1"
        assert settings.sql_gen_model == "gpt-4o"
        assert settings.max_sql_attempts == 3

    def test_subclass_of_base_settings(self) -> None:
        assert issubclass(Settings, BaseSettings)

    def test_has_oracle_credentials(self) -> None:
        assert _settings(
            oracle_dsn="db:1521/orcl",
            oracle_user="scott",
            oracle_password="tiger",
        ).has_oracle_credentials

        partial = _settings(oracle_dsn="db:1521/orcl", oracle_user="scott", oracle_password="")
        assert not partial.has_oracle_credentials

    def test_has_llm_credentials(self) -> None:
        assert _settings(sql_gen_api_key="key").has_llm_credentials
        assert not _settings(sql_gen_api_key="").has_llm_credentials

    def test_has_langfuse_credentials(self) -> None:
        assert _settings(
            langfuse_host="https://lf.local",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        ).has_langfuse_credentials

        missing_host = _settings(
            langfuse_host="",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        )
        assert not missing_host.has_langfuse_credentials


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    get_settings.cache_clear()
    assert first is second


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQL_GEN_MODEL", "gpt-test")
    monkeypatch.setenv("MAX_SQL_ATTEMPTS", "5")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.sql_gen_model == "gpt-test"
        assert settings.max_sql_attempts == 5
    finally:
        get_settings.cache_clear()


def _settings(**overrides: str | int | bool) -> Settings:
    defaults: dict[str, str | int | bool] = {
        "sql_gen_base_url": "https://llm.nalits.com/v1",
        "sql_gen_model": "gpt-4o",
        "sql_gen_api_key": "",
        "oracle_dsn": "",
        "oracle_user": "",
        "oracle_password": "",
        "langfuse_host": "",
        "langfuse_public_key": "",
        "langfuse_secret_key": "",
        "langfuse_enabled": False,
        "max_sql_attempts": 3,
    }
    defaults.update(overrides)
    return Settings.model_validate(defaults)
