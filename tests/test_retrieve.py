"""Tests for the Retrieve agent: registration, factual results, and CLI."""

from typing import Any

import pytest

from agents.common.config import Settings
from agents.common.db import QueryResult
from agents.common.llm import LLMError
from agents.registry import AgentCategory, get_agent, get_registrations, route
from agents.retrieve import cli
from agents.retrieve.graph import build_graph, run_agent
from tests.doubles import FakeLLM, FakeOracle, make_context

_QUERY = "Show total sales by region"
GOOD_SQL = "SELECT region, SUM(total) FROM SALES.VW_SALES_SUMMARY GROUP BY region"
ANSWER = "Total sales for the West region were 100 units."

SAMPLE_SCHEMA: dict[str, Any] = {
    "SALES": {
        "tables": {
            "VW_SALES_SUMMARY": {
                "type": "VIEW",
                "description": "Sales summary",
                "columns": {
                    "REGION": {"type": "VARCHAR2", "description": "Region"},
                    "TOTAL": {"type": "NUMBER", "description": "Total"},
                },
            }
        }
    }
}


def _context() -> Any:
    return make_context(
        llm=FakeLLM([GOOD_SQL, ANSWER]),
        connection=FakeOracle(
            fetch_schema=SAMPLE_SCHEMA,
            result=QueryResult(columns=["REGION"], rows=[["West", 100]]),
        ),
    )


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


def test_retrieve_is_registered_for_retrieve_category() -> None:
    registration = get_agent(AgentCategory.RETRIEVE)
    assert registration.name == "retrieve"
    assert registration.handler is run_agent


def test_sql_generator_is_not_registered() -> None:
    names = {registration.name for registration in get_registrations()}
    assert "retrieve" in names
    assert "sql_generator" not in names


def test_build_graph_returns_compiled_graph() -> None:
    graph = build_graph(make_context())
    assert callable(graph.invoke)
    assert "check_db_connection" in graph.nodes


def test_run_agent_returns_factual_data() -> None:
    state = run_agent(_QUERY, _context())
    assert state["error"] is None
    assert state["done"] is True
    assert state["query_columns"] == ["REGION"]
    assert state["query_rows"] == [["West", 100]]
    assert state["sql_query"] == GOOD_SQL
    assert state["answer"] == ANSWER


def test_run_agent_falls_back_to_template_answer_on_llm_error() -> None:
    context = make_context(
        llm=FakeLLM([GOOD_SQL]),
        connection=FakeOracle(
            fetch_schema=SAMPLE_SCHEMA,
            result=QueryResult(columns=["REGION"], rows=[["West", 100]]),
        ),
    )
    state = run_agent(_QUERY, context)
    assert state["error"] is None
    assert "REGION: West" in (state["answer"] or "")
    assert "returned 1 row(s)" in (state["answer"] or "")


def test_run_agent_no_row_no_answer() -> None:
    context = make_context(
        llm=FakeLLM([GOOD_SQL]),
        connection=FakeOracle(
            fetch_schema=SAMPLE_SCHEMA,
            result=QueryResult(columns=["REGION"], rows=[]),
        ),
    )
    state = run_agent(_QUERY, context)
    assert state["done"] is True
    assert state["query_rows"] == []
    assert state["answer"] is None


def test_route_retrieve_returns_factual_data() -> None:
    state = route(AgentCategory.RETRIEVE, _QUERY, _context())
    assert state["error"] is None
    assert state["query_rows"] == [["West", 100]]
    assert state["answer"] == ANSWER


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
    context = _context()
    monkeypatch.setattr(cli, "build_context", lambda settings: context)
    assert cli.main([_QUERY]) == 0
    output = capsys.readouterr().out
    assert "West" in output


def test_main_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: _settings())
    context = make_context(connection=FakeOracle(connected=False))
    monkeypatch.setattr(cli, "build_context", lambda settings: context)
    assert cli.main([_QUERY]) == 1


def test_main_module_is_importable() -> None:
    import agents.retrieve.__main__  # noqa: F401
