"""Placeholder handler for the Act agent (registered for routing)."""

from agents.common.placeholder import make_placeholder
from agents.registry import AgentHandler

run_agent: AgentHandler = make_placeholder("act")

__all__ = ["run_agent"]
