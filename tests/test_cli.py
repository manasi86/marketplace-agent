"""Tests for the standalone CLI."""

import pytest

from agents.common.config import Settings
from agents.common.db import QueryResult
from agents.common.llm import LLMError
from agents.sql_generator import cli
from tests.doubles import FakeLLM, FakeOracle, make_context

_QUERY = "Show total sales by region"

INTENT_JSON = '{"intent": "aggregate sales", "entities": {}, "schema_hint": "SALES"}'
GOOD_SQL = "SELECT region, SUM(total) FROM SALES.VW_SALES_SUMMARY GROUP BY region"


def _settings(**overrides: str | int | bool) -> Settings:
    defaults: dict[str, str | int | bool] = {
        "sql_gen_api_key": "test-key",
        "sql_gen_base_url": "https://llm.nalits.com/v1",
        "oracle_dsn": "db:1521/orcl",
        "oracle_user": "scott",
        "oracle_password": "tiger",
        "langfuse_enabled": False,
    }
    defaults.update(overrides)
    return Settings.model_validate(defaults)


def test_parse_args() -> None:
    args = cli.parse_args([_QUERY])
    assert args.query == _QUERY


def test_parse_args_missing_query() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args([])


def test_main_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(sql_gen_api_key=""))
    assert cli.main([_QUERY]) == 2
    assert "SQL_GEN_API_KEY" in capsys.readouterr().err


def test_main_missing_oracle_credentials(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: _settings(oracle_dsn="", oracle_user="", oracle_password=""),
    )
    assert cli.main([_QUERY]) == 2
    assert "ORACLE_DSN" in capsys.readouterr().err


def test_main_llm_construct_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings())

    def raise_llm_error(settings: Settings) -> object:
        del settings
        raise LLMError("Missing LLM API key.")

    monkeypatch.setattr(cli, "build_context", raise_llm_error)
    assert cli.main([_QUERY]) == 2
    assert "Missing LLM API key" in capsys.readouterr().err


def test_main_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings())
    context = make_context(
        llm=FakeLLM([INTENT_JSON, GOOD_SQL]),
        connection=FakeOracle(result=QueryResult(columns=["REGION"], rows=[["West", 100]])),
    )
    monkeypatch.setattr(cli, "build_context", lambda settings: context)
    assert cli.main([_QUERY]) == 0
    output = capsys.readouterr().out
    assert "Query Results" in output
    assert "West" in output


def test_main_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings())
    context = make_context(connection=FakeOracle(connected=False))
    monkeypatch.setattr(cli, "build_context", lambda settings: context)
    assert cli.main([_QUERY]) == 1


def test_main_module_is_importable() -> None:
    import agents.sql_generator.__main__ as main_module

    assert main_module is not None
