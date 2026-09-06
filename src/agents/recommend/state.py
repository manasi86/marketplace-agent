"""State threaded through the Recommend agent's interactive planning flow.

The recommend agent is conversational: it may prompt the user for a missing
goal, present a prioritised plan, and accept revisions until the user approves.
This state records where that conversation is and everything the downstream
Observe and Act agents need to act on the approved plan.
"""

from typing import Any, Literal

from agents.common.state import BaseAgentState, initial_state

RecommendStatus = Literal[
    "awaiting_goal",
    "awaiting_approval",
    "approved",
    "handed_to_act",
]


class RecommendState(BaseAgentState):
    """State threaded through the recommend agent, adding the plan context."""

    user_query: str
    goal: str | None
    plan: str | None
    observe_duration: str | None
    data_summary: str | None
    queries_executed: list[str]
    approved: bool
    needs_goal: bool
    status: RecommendStatus
    _last_parsed: dict[str, Any]


def initial_recommend_state(user_query: str) -> RecommendState:
    """Build a fresh RecommendState for a new planning conversation."""
    base = initial_state(user_query)
    return RecommendState(
        user_query=base["user_query"],
        logs=base["logs"],
        error=base["error"],
        done=base["done"],
        goal=None,
        plan=None,
        observe_duration=None,
        data_summary=None,
        queries_executed=[],
        approved=False,
        needs_goal=False,
        status="awaiting_goal",
        _last_parsed={},
    )
