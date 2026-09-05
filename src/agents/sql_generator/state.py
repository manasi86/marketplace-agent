"""Graph state for the shared SQL generator pipeline."""

from operator import add
from typing import Annotated, Any

from agents.common.state import BaseAgentState


class SqlGeneratorState(BaseAgentState):
    """State threaded through the SQL generator pipeline."""

    user_query: str
    schema_hint: str | None
    db_connected: bool
    semantic_context: str
    semantic_metadata: dict[str, Any]
    sql_query: str
    validation_error: str | None
    attempt_count: int
    max_attempts: int
    query_columns: list[str] | None
    query_rows: list[list[Any]] | None
    execution_time_ms: float | None
    answer: str | None
    logs: Annotated[list[tuple[str, str]], add]
    error: str | None
    done: bool


def initial_state(user_query: str, max_attempts: int) -> SqlGeneratorState:
    """Build a fresh SqlGeneratorState with sensible defaults for a new run."""
    return SqlGeneratorState(
        user_query=user_query,
        schema_hint=None,
        db_connected=False,
        semantic_context="",
        semantic_metadata={},
        sql_query="",
        validation_error=None,
        attempt_count=0,
        max_attempts=max_attempts,
        query_columns=None,
        query_rows=None,
        execution_time_ms=None,
        answer=None,
        logs=[],
        error=None,
        done=False,
    )
