"""Agent registry: categorisation, registration and routing.

The coordinator classifies a user query into one of :class:`AgentCategory`
and then routes it to the registered handler for that category. Future
agents register themselves here so new capabilities plug in without
touching the coordinator.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agents.common.context import AgentContext

AgentHandler = Callable[[str, AgentContext], Any]

_DEFAULT_CATEGORY = "retrieve"


class AgentCategory(str, Enum):
    """The four intent categories understood by the coordinator."""

    ACT = "act"
    RETRIEVE = "retrieve"
    RECOMMEND = "recommend"
    OBSERVE = "observe"


@dataclass(frozen=True)
class AgentRegistration:
    """Descriptive metadata plus the runnable handler for one agent."""

    category: AgentCategory
    name: str
    description: str
    handler: AgentHandler


class UnknownCategoryError(KeyError):
    """Raised when routing to a category with no registered agent."""


class MissingAgentError(KeyError):
    """Raised when routing to a category that has no handler."""


_AGENTS: dict[AgentCategory, AgentRegistration] = {}


def register(registration: AgentRegistration) -> AgentRegistration:
    """Register an agent under its category, replacing any previous entry."""
    _AGENTS[registration.category] = registration
    return registration


def unregister(category: AgentCategory) -> None:
    """Remove the registered agent for ``category`` (used by tests)."""
    _AGENTS.pop(category, None)


def get_registrations() -> tuple[AgentRegistration, ...]:
    """Return all registered agents in category order."""
    return tuple(_AGENTS[category] for category in AgentCategory if category in _AGENTS)


def get_agent(category: AgentCategory) -> AgentRegistration:
    """Return the registered agent for ``category`` or raise ``MissingAgentError``."""
    try:
        return _AGENTS[category]
    except KeyError as exc:
        raise MissingAgentError(f"No agent registered for category {category.value!r}.") from exc


def get_agent_for_raw(raw: str) -> AgentCategory:
    """Map a raw LLM string to the closest category; defaults to RETRIEVE."""
    try:
        return AgentCategory(raw.strip().lower())
    except ValueError:
        return AgentCategory(_DEFAULT_CATEGORY)


def route(category: AgentCategory, user_query: str, context: AgentContext) -> Any:
    """Route ``user_query`` to the registered agent for ``category``.

    The agent's state dict is returned as-is; the coordinator layer is
    responsible for merging classification metadata on top.
    """
    return get_agent(category).handler(user_query, context)


def route_raw(raw: str, user_query: str, context: AgentContext) -> Any:
    """Classify from a raw category label then route, raising on unknowns."""
    category = get_agent_for_raw(raw)
    if category not in _AGENTS:
        raise UnknownCategoryError(f"No agent registered for category {category.value!r}.")
    return route(category, user_query, context)


def clear_registry() -> None:
    """Remove every registered agent (used by tests to reset state)."""
    _AGENTS.clear()
