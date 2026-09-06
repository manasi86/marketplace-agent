"""Tests for the Rich-based display helpers."""

import io
from typing import Any, cast

import pytest
from rich.console import Console

from agents.common.display import (
    Results,
    _run_pager,
    build_answer_panel,
    build_failure_panel,
    build_log_panels,
    build_result_table,
    print_agent_output,
    print_result_table_paginated,
)
from agents.sql_generator.display import NODE_TITLES as SQL_NODE_TITLES
from agents.sql_generator.display import build_results
from agents.sql_generator.display import print_agent_output as sql_print
from agents.sql_generator.state import SqlGeneratorState, initial_state

NODE_TITLES = {"generate_sql": "3. Generate SQL", "validate_sql": "4. Validate SQL"}


def _state(**overrides: Any) -> SqlGeneratorState:
    state = initial_state("Show sales", 3)
    state.update(cast(Any, overrides))
    return state


def _console() -> tuple[io.StringIO, Console]:
    buffer = io.StringIO()
    return buffer, Console(file=buffer, width=200)


def test_build_log_panels_groups_by_node() -> None:
    state = _state(
        logs=[
            ("generate_sql", "first log"),
            ("validate_sql", "second log"),
            ("generate_sql", "third log"),
        ]
    )
    panels = build_log_panels(state, NODE_TITLES)
    assert [panel.title for panel in panels] == ["3. Generate SQL", "4. Validate SQL"]
    assert "first log" in str(panels[0].renderable)
    assert "third log" in str(panels[0].renderable)
    assert "second log" not in str(panels[0].renderable)
    assert "second log" in str(panels[1].renderable)


def test_build_log_panels_defaults_to_node_names() -> None:
    panels = build_log_panels(_state(logs=[("custom_node", "a message")]))
    assert panels[0].title == "custom_node"


def test_build_log_panels_empty_logs() -> None:
    assert build_log_panels(_state(logs=[])) == []


def test_build_failure_panel() -> None:
    rendered = str(build_failure_panel("ORA-00942: table does not exist").renderable)
    assert "ORA-00942" in rendered


def test_build_answer_panel() -> None:
    panel = build_answer_panel("  Total sales were 100 units.  ")
    assert panel.title == "Answer"
    assert "Total sales were 100 units." in str(panel.renderable)


def test_build_result_table_columns_rows_and_caption() -> None:
    results = Results(
        columns=["REGION", "TOTAL"],
        rows=[["West", 100], ["East", 200]],
        execution_time_ms=12.3,
        attempt_count=1,
        sql_query="SELECT region, SUM(total) FROM SALES.VW_SALES_SUMMARY GROUP BY region",
    )
    table = build_result_table(results)
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
    results = Results(
        columns=["A"],
        rows=[["1"]],
        execution_time_ms=1.5,
        attempt_count=0,
        sql_query=long_sql,
    )
    caption = str(build_result_table(results).caption or "")
    assert "..." in caption


def test_build_result_table_empty_result() -> None:
    results = Results(execution_time_ms=None, attempt_count=0, sql_query="")
    table = build_result_table(results)
    assert table.columns == []
    caption = str(table.caption or "")
    assert "No timing recorded" in caption
    assert "SQL:" not in caption


def test_print_agent_output_success() -> None:
    state = _state(logs=[("generate_sql", "step one")])
    buffer, console = _console()
    print_agent_output(
        state,
        node_titles=NODE_TITLES,
        results=Results(columns=["REGION"], rows=[["West"]], execution_time_ms=5.0),
        console=console,
    )
    output = buffer.getvalue()
    assert "3. Generate SQL" in output
    assert "step one" in output
    assert "West" in output
    assert "REGION" in output


def test_print_agent_output_with_answer() -> None:
    state = _state()
    buffer, console = _console()
    print_agent_output(
        state,
        node_titles=NODE_TITLES,
        results=Results(columns=["REGION"], rows=[["West"]], execution_time_ms=5.0),
        answer="Total sales for the West region were 100 units.",
        console=console,
    )
    output = buffer.getvalue()
    assert "Answer" in output
    assert "Total sales for the West region were 100 units." in output
    assert "West" in output


def test_print_agent_output_failure() -> None:
    state = _state()
    buffer, console = _console()
    print_agent_output(
        state,
        node_titles=NODE_TITLES,
        failure_message="connection refused",
        console=console,
    )
    assert "Failure" in buffer.getvalue()
    assert "connection refused" in buffer.getvalue()


def test_print_agent_output_no_result() -> None:
    buffer, console = _console()
    print_agent_output(_state(), console=console)
    assert "without producing a result" in buffer.getvalue()


def test_sql_build_results() -> None:
    state = _state(
        query_columns=["REGION"],
        query_rows=[["West"]],
        execution_time_ms=12.0,
        attempt_count=1,
        sql_query="SELECT region FROM SALES.VW_SALES_SUMMARY",
    )
    results = build_results(state)
    assert results.columns == ["REGION"]
    assert results.rows == [["West"]]
    assert results.execution_time_ms == 12.0
    assert results.attempt_count == 1
    assert "SELECT region" in results.sql_query


def test_sql_build_results_defaults() -> None:
    results = build_results(_state())
    assert results.columns == []
    assert results.rows == []
    assert results.execution_time_ms is None
    assert results.attempt_count == 0
    assert results.sql_query == ""


def test_sql_print_agent_output_uses_titles() -> None:
    state = _state(
        query_columns=["REGION"],
        query_rows=[["West"]],
        logs=[("generate_sql", "step one")],
        execution_time_ms=5.0,
    )
    buffer, console = _console()
    sql_print(state, console)
    output = buffer.getvalue()
    assert SQL_NODE_TITLES["generate_sql"] in output
    assert "REGION" in output
    assert "West" in output


def test_sql_print_agent_output_validation_error() -> None:
    state = _state(validation_error="ORA-00904: bad column")
    buffer, console = _console()
    sql_print(state, console)
    assert "ORA-00904" in buffer.getvalue()


def test_sql_print_agent_output_no_result() -> None:
    state = _state(logs=[])
    buffer, console = _console()
    sql_print(state, console)
    assert "without producing a result" in buffer.getvalue()


def _many_rows(count: int) -> Results:
    return Results(
        columns=["ID"],
        rows=[[str(index)] for index in range(1, count + 1)],
        execution_time_ms=5.0,
        attempt_count=1,
        sql_query="SELECT id FROM SALES.VW_SALES_SUMMARY",
    )


def _normalize(text: str) -> str:
    """Flatten whitespace/newlines so Rich-wrapped captions match literally."""
    return " ".join(text.split())


def test_non_interactive_large_result_prints_first_page_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agents.common.display._is_interactive", lambda: False)
    buffer, console = _console()
    print_result_table_paginated(
        _many_rows(120),
        console,
        page_size=50,
    )
    output = _normalize(buffer.getvalue())
    assert "50" in output
    assert "51" not in output
    assert "of 120" in output


def test_non_interactive_large_result_does_not_show_last_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agents.common.display._is_interactive", lambda: False)
    buffer, console = _console()
    print_result_table_paginated(_many_rows(120), console, page_size=50)
    assert "101" not in _normalize(buffer.getvalue())


def test_non_interactive_small_result_prints_all_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agents.common.display._is_interactive", lambda: True)
    buffer, console = _console()
    print_result_table_paginated(_many_rows(3), console, page_size=50, input_fn=lambda _: "q")
    assert "of 3" in _normalize(buffer.getvalue())
    assert "3" in _normalize(buffer.getvalue())


def test_pager_next_and_quit() -> None:
    buffer, console = _console()
    commands = iter(["n", "q"])
    _run_pager(_many_rows(120), console, 50, input_fn=lambda _: next(commands))
    output = _normalize(buffer.getvalue())
    assert "51" in output
    assert "100" in output
    assert "101" not in output


def test_pager_prev_and_jump() -> None:
    buffer, console = _console()
    commands = iter(["j 3", "p", "q"])
    _run_pager(_many_rows(120), console, 50, input_fn=lambda _: next(commands))
    output = _normalize(buffer.getvalue())
    assert "101" in output
    assert "119" in output
    assert "51" in output
    assert "100" in output


def test_pager_jump_out_of_range_keeps_page() -> None:
    buffer, console = _console()
    commands = iter(["j 99", "q"])
    _run_pager(_many_rows(120), console, 50, input_fn=lambda _: next(commands))
    output = _normalize(buffer.getvalue())
    assert "50" in output
    assert "51" not in output


def test_pager_eof_returns() -> None:
    buffer, console = _console()

    def raise_eof(_: str) -> str:
        raise EOFError

    _run_pager(_many_rows(120), console, 50, input_fn=raise_eof)
    assert "50" in _normalize(buffer.getvalue())
    assert "51" not in _normalize(buffer.getvalue())


def test_pager_unknown_and_invalid_jump_keep_page() -> None:
    buffer, console = _console()
    commands = iter(["x", "j not-a-number", "q"])
    _run_pager(_many_rows(120), console, 50, input_fn=lambda _: next(commands))
    output = _normalize(buffer.getvalue())
    assert "50" in output
    assert "51" not in output


def test_print_result_table_paginated_uses_real_is_interactive() -> None:
    buffer, console = _console()
    print_result_table_paginated(
        _many_rows(120),
        console,
        page_size=50,
        input_fn=lambda _: "q",
    )
    output = _normalize(buffer.getvalue())
    assert "50" in output
    assert "51" not in output


def test_print_result_table_paginated_starts_pager_when_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agents.common.display._is_interactive", lambda: True)
    buffer, console = _console()
    print_result_table_paginated(
        _many_rows(120),
        console,
        page_size=50,
        input_fn=lambda _: "q",
    )
    output = _normalize(buffer.getvalue())
    assert "50" in output
    assert "51" not in output


def test_build_result_table_page_caption() -> None:
    results = Results(
        columns=["ID"],
        rows=[["1"], ["2"]],
        execution_time_ms=5.0,
        attempt_count=0,
        sql_query="",
    )
    caption = str(build_result_table(results, page_start=51, total_rows=120).caption or "")
    assert "showing rows 51-52 of 120" in caption
