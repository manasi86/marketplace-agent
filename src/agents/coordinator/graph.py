"""LangGraph wiring for the coordinator agent.

The coordinator classifies a query into a single :class:`AgentCategory` and
then routes it to the registered handler for that category via the registry.
"""

import logging
from typing import Any, cast

from langgraph.graph import END, StateGraph

from agents.common.context import AgentContext
from agents.common.observability import log_step, observe_run, observe_step
from agents.coordinator.nodes import build_nodes
from agents.coordinator.state import CoordinatorState, initial_coordinator_state
from agents.registry import route

logger = logging.getLogger(__name__)


def build_graph(context: AgentContext) -> Any:
    """Compile the coordinator's StateGraph with its classification node."""
    nodes = build_nodes(context)
    graph = StateGraph(CoordinatorState)
    graph.add_node(  # type: ignore[call-overload]
        "classify_intent",
        observe_step("classify_intent")(log_step("classify_intent")(nodes["classify_intent"])),
    )
    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", END)
    return graph.compile()


@observe_run("coordinator_agent")
def classify_query(user_query: str, context: AgentContext) -> CoordinatorState:
    """Run classification only, returning the parsed category in state."""
    logger.info("Coordinator classifying query: %s", user_query)
    graph = build_graph(context)
    state = initial_coordinator_state(user_query)
    return cast(CoordinatorState, graph.invoke(state))


@observe_run("coordinator_agent")
def run_agent(user_query: str, context: AgentContext) -> dict[str, Any]:
    """Classify the query, route it to the matching agent and merge its result.

    Returns a single flat state carrying the classification metadata on top of
    whatever the routed agent produced, so callers render one coherent output.
    """
    logger.info("Coordinator routing query: %s", user_query)
    classified = classify_query(user_query, context)
    category = classified["category"]
    assert category is not None
    result = route(category, user_query, context)
    merged = {
        **classified,
        **result,
        "category": category,
        "logs": [
            *classified.get("logs", []),
            *result.get("logs", []),
        ],
    }
    if result.get("error") is not None:
        logger.error("Coordinator run finished with an error: %s", result["error"])
    else:
        logger.info("Coordinator routed '%s' to %r.", user_query, category.value)
    return merged
