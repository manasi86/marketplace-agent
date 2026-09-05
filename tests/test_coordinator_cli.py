"""Tests for the coordinator CLI."""

from typing import Any

import pytest

from agents.common.config import Settings
from agents.common.llm import LLMError
from agents.coordinator import cli
from agents.registry import AgentCategory
from tests.doubles import FakeLLM, make_context

_QUERY = "Show total sales by region"
INTENT_JSON = '{"intent": "aggregate sales", "entities": {}, "schema_hint": "SALES"}'
GOOD_SQL = "SELECT region, SUM(total) FROM SALES.VW_SALES_SUMMARY GROUP BY region"


def _settings(**overrides: str | int | bool) -> Settings:
    defaults: dict[str, str | int | bool] = {
        "sql_gen_api_key": "test-key",
        "sql_gen_base_url": "https://llm.nalits.com/v1",
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


def test_main_success_retrieve_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings())
    context = make_context(llm=FakeLLM([]))
    state = {
        "user_query": _QUERY,
        "category": AgentCategory.RETRIEVE,
        "logs": [("classify_intent", "Classified as 'retrieve'")],
        "error": None,
        "done": True,
        "query_columns": ["REGION"],
        "query_rows": [["West", 100]],
        "execution_time_ms": 3.5,
        "sql_query": GOOD_SQL,
        "attempt_count": 0,
    }
    monkeypatch.setattr(cli, "build_context", lambda settings: context)
    monkeypatch.setattr(cli, "run_agent", lambda query, ctx: state)
    assert cli.main([_QUERY]) == 0
    output = capsys.readouterr().out
    assert "Classified as 'retrieve'" in output
    assert "REGION" in output
    assert "West" in output


def test_main_error_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings())
    state = {
        "user_query": _QUERY,
        "category": AgentCategory.ACT,
        "logs": [("placeholder", "Agent 'act' is not yet implemented.")],
        "error": "Agent 'act' is not yet implemented.",
        "done": True,
    }
    monkeypatch.setattr(cli, "build_context", lambda settings: make_context(llm=FakeLLM([])))
    monkeypatch.setattr(cli, "run_agent", lambda query, ctx: state)
    assert cli.main([_QUERY]) == 1
    assert "Failure" in capsys.readouterr().out


def test_main_module_is_importable() -> None:
    import agents.coordinator.__main__ as main_module

    assert main_module is not None


def test_build_results_ignores_non_tabular_state() -> None:
    state: dict[str, Any] = {"user_query": _QUERY, "logs": [], "error": None, "done": True}
    assert cli._build_results(state) is None


def test_build_results_extracts_table() -> None:
    state: dict[str, Any] = {
        "query_columns": ["REGION"],
        "query_rows": [["West"]],
        "execution_time_ms": 2.0,
        "sql_query": GOOD_SQL,
        "attempt_count": 1,
    }
    results = cli._build_results(state)
    assert results is not None
    assert results.columns == ["REGION"]
    assert results.rows == [["West"]]
    assert results.execution_time_ms == 2.0
    assert results.attempt_count == 1


def test_build_results_handles_non_numeric_timing() -> None:
    state: dict[str, Any] = {
        "query_rows": [["x"]],
        "execution_time_ms": "fast",
        "attempt_count": None,
    }
    results = cli._build_results(state)
    assert results is not None
    assert results.execution_time_ms is None
    assert results.attempt_count == 0


def test_build_results_handles_unexpected_column_types() -> None:
    state: dict[str, Any] = {"query_rows": "not-a-list", "query_columns": "not-a-list"}
    results = cli._build_results(state)
    assert results is not None
    assert results.columns == []
    assert results.rows == []
