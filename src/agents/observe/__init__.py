"""Observe agent: registered placeholder for future monitoring capabilities."""

from agents.observe.agent import run_agent
from agents.registry import AgentCategory, AgentRegistration

REGISTRATION = AgentRegistration(
    category=AgentCategory.OBSERVE,
    name="observe",
    description="Monitors, tracks or watches resources over time and reports status.",
    handler=run_agent,
)

__all__ = ["REGISTRATION", "run_agent"]
