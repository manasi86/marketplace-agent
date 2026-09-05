"""Tests for the Rich-based display helpers."""

import io
from typing import Any, cast

from rich.console import Console

from lib.sql_generator.display import (
    build_error_panel,
    build_log_panels,
    build_result_table,
    print_agent_output,
)
from lib.sql_generator.state import AgentState, initial_state


def _state(**overrides: Any) -> AgentState:
    state = initial_state("Show sales", 3)
    state.update(cast(Any, overrides))
    return state


def test_build_result_table_columns_rows_and_caption() -> None:
    state = _state(
        query_columns=["REGION", "TOTAL"],
        query_rows=[["West", 100], ["East", 200]],
        execution_time_ms=12.3,
        attempt_count=1,
        sql_query="SELECT region, SUM(total) FROM SALES.VW_SALES_SUMMARY GROUP BY region",
    )
    table = build_result_table(state)
    assert table.title == "Query Results"
    assert [column.header for column in table.columns] == ["REGION", "TOTAL"]
    caption = table.caption
    assert caption is not None
    rendered = str(caption)
    assert "Executed in 12.3 ms" in rendered
    assert "1 generate/validate attempt(s)" in rendered
    assert "SQL: SELECT region" in rendered
    assert "..." not in rendered


def test_build_result_table_long_sql_truncated() -> None:
    long_sql = "SELECT " + "a, " * 60
    state = _state(
        query_columns=["A"],
        query_rows=[["1"]],
        execution_time_ms=1.5,
        attempt_count=0,
        sql_query=long_sql,
    )
    caption = str(build_result_table(state).caption or "")
    assert "..." in caption


def test_build_result_table_empty_result() -> None:
    state = _state(
        query_columns=[],
        query_rows=[],
        execution_time_ms=None,
        attempt_count=0,
        sql_query="",
    )
    table = build_result_table(state)
    assert table.columns == []
    caption = str(table.caption or "")
    assert "No timing recorded" in caption
    assert "SQL:" not in caption


def test_build_log_panels_groups_by_node() -> None:
    state = _state(
        logs=[
            ("understand_intent", "first log"),
            ("generate_sql", "second log"),
            ("understand_intent", "third log"),
        ]
    )
    panels = build_log_panels(state)
    assert [panel.title for panel in panels] == [
        "1. Understand Intent",
        "4. Generate SQL",
    ]
    assert "first log" in str(panels[0].renderable)
    assert "third log" in str(panels[0].renderable)
    assert "second log" not in str(panels[0].renderable)
    assert "second log" in str(panels[1].renderable)


def test_build_log_panels_empty_logs() -> None:
    assert build_log_panels(_state(logs=[])) == []


def test_build_log_panels_unknown_node_uses_name() -> None:
    panels = build_log_panels(_state(logs=[("custom_node", "a message")]))
    assert panels[0].title == "custom_node"


def test_build_error_panel_with_error() -> None:
    state = _state(error="ORA-00942: table does not exist")
    rendered = str(build_error_panel(state).renderable)
    assert "ORA-00942" in rendered


def test_build_error_panel_with_validation_error_only() -> None:
    state = _state(validation_error="ORA-00904: invalid identifier")
    rendered = str(build_error_panel(state).renderable)
    assert "ORA-00904" in rendered


def test_build_error_panel_unknown() -> None:
    rendered = str(build_error_panel(_state()).renderable)
    assert "Unknown error" in rendered


def test_print_agent_output_success() -> None:
    state = _state(
        query_columns=["REGION"],
        query_rows=[["West"]],
        logs=[("understand_intent", "step one")],
        execution_time_ms=5.0,
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=200)
    print_agent_output(state, console)
    output = buffer.getvalue()
    assert "1. Understand Intent" in output
    assert "step one" in output
    assert "West" in output
    assert "REGION" in output


def test_print_agent_output_error() -> None:
    state = _state(error="connection refused")
    buffer = io.StringIO()
    print_agent_output(state, Console(file=buffer, width=200))
    assert "Failure" in buffer.getvalue()


def test_print_agent_output_validation_error() -> None:
    state = _state(validation_error="ORA-00904: bad column")
    buffer = io.StringIO()
    print_agent_output(state, Console(file=buffer, width=200))
    assert "ORA-00904" in buffer.getvalue()


def test_print_agent_output_no_result() -> None:
    state = _state()
    buffer = io.StringIO()
    print_agent_output(state, Console(file=buffer, width=200))
    assert "without producing a result" in buffer.getvalue()
