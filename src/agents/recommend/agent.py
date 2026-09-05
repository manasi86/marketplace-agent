"""Placeholder handler for the Recommend agent (registered for routing)."""

from agents.common.placeholder import make_placeholder
from agents.registry import AgentHandler

run_agent: AgentHandler = make_placeholder("recommend")

__all__ = ["run_agent"]
