"""Act agent: registered placeholder for future action-taking capabilities."""

from agents.act.agent import run_agent
from agents.registry import AgentCategory, AgentRegistration

REGISTRATION = AgentRegistration(
    category=AgentCategory.ACT,
    name="act",
    description="Performs actions or mutations (create, update, delete, apply) on systems.",
    handler=run_agent,
)

__all__ = ["REGISTRATION", "run_agent"]
