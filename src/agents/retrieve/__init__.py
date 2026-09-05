"""Retrieve agent: answers factual data questions by running the shared SQL pipeline."""

from agents.registry import AgentCategory, AgentRegistration
from agents.retrieve.graph import run_agent

REGISTRATION = AgentRegistration(
    category=AgentCategory.RETRIEVE,
    name="retrieve",
    description=(
        "Returns factual data for natural-language questions by running the "
        "shared SQL generator pipeline and reporting the results back."
    ),
    handler=run_agent,
)

__all__ = ["REGISTRATION", "run_agent"]
