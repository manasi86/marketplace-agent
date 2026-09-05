"""Output helpers for the Retrieve agent: English answer plus results table."""

from rich.console import Console

from agents.common.display import print_agent_output as print_common_output
from agents.sql_generator.display import NODE_TITLES, build_results
from agents.sql_generator.state import SqlGeneratorState

__all__ = ["NODE_TITLES", "build_results"]


def print_agent_output(state: SqlGeneratorState, console: Console | None = None) -> None:
    """Print the agent's English answer and the underlying results table."""
    print_common_output(
        state,
        node_titles=NODE_TITLES,
        results=build_results(state) if state.get("query_rows") is not None else None,
        failure_message=state.get("error") or state.get("validation_error"),
        answer=state.get("answer"),
        console=console,
    )
