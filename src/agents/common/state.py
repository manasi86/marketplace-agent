"""Shared graph state primitives used by every agent."""

from operator import add
from typing import Annotated, TypedDict


class BaseAgentState(TypedDict):
    """Mutable state threaded through every agent pipeline.

    Subclasses (e.g. ``CoordinatorState``, the SQL generator's
    ``SqlGeneratorState``) extend this with their own category- or
    domain-specific fields.
    """

    user_query: str
    logs: Annotated[list[tuple[str, str]], add]
    error: str | None
    done: bool


def initial_state(user_query: str) -> BaseAgentState:
    """Build a fresh base state for a new run."""
    return BaseAgentState(
        user_query=user_query,
        logs=[],
        error=None,
        done=False,
    )
