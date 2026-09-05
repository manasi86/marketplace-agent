"""Shared graph state for the SQL generator agent."""

from datetime import date
from operator import add
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict):
    """Mutable state threaded through every node in the LangGraph pipeline."""

    user_query: str
    intent: str
    entities: dict[str, Any]
    schema_hint: str | None
    date_start: date | None
    date_end: date | None
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
    logs: Annotated[list[tuple[str, str]], add]
    error: str | None
    done: bool


def initial_state(user_query: str, max_attempts: int) -> AgentState:
    """Build a fresh AgentState with sensible defaults for a new run."""
    return AgentState(
        user_query=user_query,
        intent="",
        entities={},
        schema_hint=None,
        date_start=None,
        date_end=None,
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
        logs=[],
        error=None,
        done=False,
    )
