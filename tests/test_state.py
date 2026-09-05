"""Tests for the graph state model."""

from agents.sql_generator.state import initial_state


def test_initial_state_defaults() -> None:
    state = initial_state("Show sales", max_attempts=3)

    assert isinstance(state, dict)
    assert state["user_query"] == "Show sales"
    assert state["max_attempts"] == 3
    assert state["attempt_count"] == 0
    assert state["logs"] == []
    assert state["done"] is False
    assert state["error"] is None
    assert state["validation_error"] is None
    assert state["sql_query"] == ""
    assert state["semantic_context"] == ""
    assert state["semantic_metadata"] == {}
    assert state["entities"] == {}
    assert state["query_rows"] is None
    assert state["query_columns"] is None
    assert state["execution_time_ms"] is None
