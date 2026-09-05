"""Tests for the agent registry and routing."""

from typing import Any

import pytest

from agents.common.context import AgentContext
from agents.registry import (
    AgentCategory,
    AgentRegistration,
    MissingAgentError,
    UnknownCategoryError,
    clear_registry,
    get_agent,
    get_agent_for_raw,
    get_registrations,
    register,
    route,
    route_raw,
    unregister,
)


def _handler(user_query: str, context: AgentContext) -> dict[str, Any]:
    del context
    return {"user_query": user_query, "done": True, "error": None, "logs": []}


def _registration(category: AgentCategory, name: str | None = None) -> AgentRegistration:
    return AgentRegistration(
        category=category,
        name=name or category.value,
        description=f"{category.value} agent",
        handler=_handler,
    )


@pytest.fixture(autouse=True)
def _restore_registry() -> Any:
    original = get_registrations()
    clear_registry()
    yield
    clear_registry()
    for registration in original:
        register(registration)


def test_register_and_get_agent() -> None:
    registration = _registration(AgentCategory.ACT)
    register(registration)
    assert get_agent(AgentCategory.ACT) is registration


def test_register_replaces_previous() -> None:
    first = _registration(AgentCategory.ACT, "first")
    second = _registration(AgentCategory.ACT, "second")
    register(first)
    register(second)
    assert get_agent(AgentCategory.ACT) is second


def test_get_agent_missing_raises() -> None:
    with pytest.raises(MissingAgentError):
        get_agent(AgentCategory.OBSERVE)


def test_unregister_removes_agent() -> None:
    register(_registration(AgentCategory.ACT))
    unregister(AgentCategory.ACT)
    with pytest.raises(MissingAgentError):
        get_agent(AgentCategory.ACT)
    unregister(AgentCategory.RECOMMEND)


def test_get_registrations_in_category_order() -> None:
    register(_registration(AgentCategory.OBSERVE))
    register(_registration(AgentCategory.ACT))
    register(_registration(AgentCategory.RECOMMEND))
    register(_registration(AgentCategory.RETRIEVE))
    assert [registration.category for registration in get_registrations()] == [
        AgentCategory.ACT,
        AgentCategory.RETRIEVE,
        AgentCategory.RECOMMEND,
        AgentCategory.OBSERVE,
    ]


def test_clear_registry() -> None:
    register(_registration(AgentCategory.ACT))
    clear_registry()
    assert get_registrations() == ()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("act", AgentCategory.ACT),
        ("RETRIEVE", AgentCategory.RETRIEVE),
        ("observing now", AgentCategory.RETRIEVE),
    ],
)
def test_get_agent_for_raw(raw: str, expected: AgentCategory) -> None:
    assert get_agent_for_raw(raw) is expected


def test_route_invokes_handler() -> None:
    register(_registration(AgentCategory.ACT))
    context = object()
    result = route(AgentCategory.ACT, "do something", context)  # type: ignore[arg-type]
    assert result["done"] is True


def test_route_unregistered_raises() -> None:
    with pytest.raises(MissingAgentError):
        route(AgentCategory.ACT, "do something", None)  # type: ignore[arg-type]


def test_route_raw_invokes_handler() -> None:
    register(_registration(AgentCategory.RECOMMEND))
    context = object()
    result = route_raw("recommend", "suggest something", context)  # type: ignore[arg-type]
    assert result["user_query"] == "suggest something"


def test_route_raw_unknown_category_raises() -> None:
    register(_registration(AgentCategory.ACT))
    with pytest.raises(UnknownCategoryError):
        route_raw("observe", "watch something", None)  # type: ignore[arg-type]
