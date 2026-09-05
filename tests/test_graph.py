"""Tests for the LangGraph pipeline wiring and full runs."""

from typing import Any

from lib.sql_generator.db import QueryResult
from lib.sql_generator.graph import build_graph, run_agent
from tests.doubles import FakeLLM, FakeOracle, make_context

INTENT_JSON = '{"intent": "aggregate sales", "entities": {}, "schema_hint": "SALES"}'
GOOD_SQL = "SELECT region, SUM(total) FROM SALES.VW_SALES_SUMMARY GROUP BY region"
BAD_SQL = "SELECT bad_column FROM doesn_exist"

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


def test_build_graph_returns_compiled_graph() -> None:
    context = make_context()
    graph = build_graph(context)
    assert callable(graph.invoke)
    node_names = set(graph.nodes)
    assert {
        "understand_intent",
        "check_db_connection",
        "discover_semantic_layer",
        "generate_sql",
        "validate_sql",
        "execute_and_display",
    } <= node_names


def test_run_agent_happy_path() -> None:
    oracle = FakeOracle(
        fetch_schema=SAMPLE_SCHEMA,
        result=QueryResult(columns=["REGION", "TOTAL"], rows=[["West", 100]]),
    )
    context = make_context(llm=FakeLLM([INTENT_JSON, GOOD_SQL]), connection=oracle)
    state = run_agent("Sum sales by region", context)
    assert state["done"] is True
    assert state["error"] is None
    assert state["validation_error"] is None
    assert state["sql_query"] == GOOD_SQL
    assert state["query_columns"] == ["REGION", "TOTAL"]
    assert state["query_rows"] == [["West", 100]]
    assert any("SQL validation: PASSED" in log[1] for log in state["logs"])
    assert oracle.explain_calls == 1


def test_run_agent_retries_then_succeeds() -> None:
    oracle = FakeOracle(
        fetch_schema=SAMPLE_SCHEMA,
        explain_failures=1,
        result=QueryResult(columns=["REGION"], rows=[["West", 100]]),
    )
    llm = FakeLLM([INTENT_JSON, BAD_SQL, GOOD_SQL])
    context = make_context(llm=llm, connection=oracle)
    state = run_agent("Sum sales", context)
    assert state["done"] is True
    assert state["error"] is None
    assert state["attempt_count"] == 1
    assert state["sql_query"] == GOOD_SQL
    assert any("fix attempt after validation error" in log[1] for log in state["logs"])
    assert oracle.explain_calls == 2


def test_run_agent_exhausts_retries() -> None:
    oracle = FakeOracle(fetch_schema=SAMPLE_SCHEMA, explain_failures=99)
    llm = FakeLLM([INTENT_JSON, BAD_SQL, BAD_SQL, BAD_SQL])
    context = make_context(llm=llm, connection=oracle)
    state = run_agent("Sum sales", context)
    assert state["done"] is True
    assert state["attempt_count"] == 3
    assert state["query_rows"] is None
    assert (state["error"] or "").startswith("SQL validation failed after 3 attempts")
    assert state["validation_error"] is not None


def test_run_agent_database_unreachable() -> None:
    context = make_context(
        llm=FakeLLM([INTENT_JSON]),
        connection=FakeOracle(connected=False),
    )
    state = run_agent("Sum sales", context)
    assert state["done"] is True
    assert state["error"] == "Could not connect to the Oracle database."
    assert state["sql_query"] == ""
    assert state["query_rows"] is None


def test_run_agent_execution_failure_sets_error() -> None:
    oracle = FakeOracle(
        fetch_schema=SAMPLE_SCHEMA,
        execute_error="ORA-06550: line 1, column 7",
    )
    context = make_context(llm=FakeLLM([INTENT_JSON, GOOD_SQL]), connection=oracle)
    state = run_agent("Sum sales", context)
    assert state["done"] is True
    assert state["error"] == "ORA-06550: line 1, column 7"
    assert state["query_rows"] is None
