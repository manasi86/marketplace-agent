"""Placeholder handler for the Observe agent (registered for routing)."""

from agents.common.placeholder import make_placeholder
from agents.registry import AgentHandler

run_agent: AgentHandler = make_placeholder("observe")

__all__ = ["run_agent"]
