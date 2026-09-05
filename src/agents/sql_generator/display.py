"""Output helpers for the shared SQL generator pipeline."""

from collections.abc import Mapping

from rich.console import Console

from agents.common.display import Results
from agents.common.display import print_agent_output as print_common_output
from agents.sql_generator.state import SqlGeneratorState

NODE_TITLES: Mapping[str, str] = {
    "check_db_connection": "1. Check DB Connection",
    "discover_semantic_layer": "2. Discover Semantic Layer",
    "generate_sql": "3. Generate SQL",
    "validate_sql": "4. Validate SQL",
    "execute_and_display": "5. Execute & Display",
}


def build_results(state: SqlGeneratorState) -> Results:
    """Build a Rich table metadata object from query result state."""
    return Results(
        columns=state.get("query_columns") or [],
        rows=state.get("query_rows") or [],
        execution_time_ms=state.get("execution_time_ms"),
        sql_query=state.get("sql_query") or "",
        attempt_count=state.get("attempt_count") or 0,
    )


def print_agent_output(state: SqlGeneratorState, console: Console | None = None) -> None:
    """Print the full agent output: per-node logs, results (or error)."""
    print_common_output(
        state,
        node_titles=NODE_TITLES,
        results=build_results(state) if state.get("query_rows") is not None else None,
        failure_message=state.get("error") or state.get("validation_error"),
        console=console,
    )
