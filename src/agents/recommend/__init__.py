"""Recommend agent: registered placeholder for future recommendation capabilities."""

from agents.recommend.agent import run_agent
from agents.registry import AgentCategory, AgentRegistration

REGISTRATION = AgentRegistration(
    category=AgentCategory.RECOMMEND,
    name="recommend",
    description="Suggests the best option for a user given criteria or preferences.",
    handler=run_agent,
)

__all__ = ["REGISTRATION", "run_agent"]
