"""Marketplace agents: a coordinator that routes queries to domain agents.

Exposes the coordinator entrypoint (``run_agent``) together with the registry
used to dispatch each query to the agent registered for its category.
"""

from agents.act import REGISTRATION as ACT_REGISTRATION
from agents.coordinator.graph import classify_query, run_agent
from agents.observe import REGISTRATION as OBSERVE_REGISTRATION
from agents.recommend import REGISTRATION as RECOMMEND_REGISTRATION
from agents.registry import (
    AgentCategory,
    AgentRegistration,
    MissingAgentError,
    UnknownCategoryError,
    register,
    route,
    route_raw,
)
from agents.retrieve import REGISTRATION as RETRIEVE_REGISTRATION

__all__ = [
    "AgentCategory",
    "AgentRegistration",
    "MissingAgentError",
    "UnknownCategoryError",
    "classify_query",
    "register",
    "route",
    "route_raw",
    "run_agent",
]

for _registration in (
    ACT_REGISTRATION,
    RECOMMEND_REGISTRATION,
    OBSERVE_REGISTRATION,
    RETRIEVE_REGISTRATION,
):
    register(_registration)
