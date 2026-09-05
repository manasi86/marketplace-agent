"""Shared factory for agents that are registered but not yet implemented."""

from typing import Any

from agents.common.context import AgentContext
from agents.registry import AgentHandler


def make_placeholder(name: str) -> AgentHandler:
    """Return a handler that records a not-yet-implemented outcome for ``name``.

    Placeholder agents occupy their category in the registry so routing is
    complete and testable; swapping in a real implementation later only
    requires replacing the registered handler.
    """

    def run(user_query: str, context: AgentContext) -> dict[str, Any]:
        del context
        message = f"Agent {name!r} is not yet implemented."
        return {
            "user_query": user_query,
            "logs": [("placeholder", message)],
            "error": message,
            "done": True,
        }

    return run
