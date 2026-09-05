"""Shared graph state for the coordinator agent."""

from agents.common.state import BaseAgentState, initial_state
from agents.registry import AgentCategory

FALLBACK_CATEGORY = AgentCategory.RETRIEVE


class CoordinatorState(BaseAgentState):
    """State threaded through the coordinator pipeline, adding classification."""

    user_query: str
    category: AgentCategory | None


def initial_coordinator_state(user_query: str) -> CoordinatorState:
    """Build a fresh CoordinatorState for a new classification run."""
    base = initial_state(user_query)
    return CoordinatorState(
        user_query=base["user_query"],
        logs=base["logs"],
        error=base["error"],
        done=base["done"],
        category=None,
    )
