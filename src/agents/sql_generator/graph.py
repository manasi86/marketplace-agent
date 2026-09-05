"""LangGraph wiring for the shared SQL generator pipeline."""

import logging
from typing import Any, cast

from langgraph.graph import END, StateGraph

from agents.common.context import AgentContext
from agents.common.observability import log_step, observe_run, observe_step
from agents.sql_generator.nodes import build_nodes
from agents.sql_generator.state import SqlGeneratorState, initial_state

logger = logging.getLogger(__name__)

NODE_NAMES = (
    "understand_intent",
    "check_db_connection",
    "discover_semantic_layer",
    "generate_sql",
    "validate_sql",
    "execute_and_display",
)


def build_graph(context: AgentContext) -> Any:
    """Compile the agent's StateGraph with traced node functions."""
    nodes = build_nodes(context)
    graph = StateGraph(SqlGeneratorState)
    for name in NODE_NAMES:
        graph.add_node(  # type: ignore[call-overload]
            name, observe_step(name)(log_step(name)(nodes[name]))
        )

    graph.set_entry_point("understand_intent")
    graph.add_edge("understand_intent", "check_db_connection")
    graph.add_conditional_edges(
        "check_db_connection",
        _route_after_connection,
        {"discover_semantic_layer": "discover_semantic_layer", END: END},
    )
    graph.add_edge("discover_semantic_layer", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")
    graph.add_conditional_edges(
        "validate_sql",
        _route_after_validation,
        {
            "generate_sql": "generate_sql",
            "execute_and_display": "execute_and_display",
            END: END,
        },
    )
    graph.add_edge("execute_and_display", END)
    return graph.compile()


@observe_run("sql_generator_agent")
def run_agent(user_query: str, context: AgentContext) -> SqlGeneratorState:
    """Execute the full SQL generator pipeline for a natural-language question."""
    logger.info("Starting agent run for query: %s", user_query)
    graph = build_graph(context)
    state = initial_state(user_query, context.settings.max_sql_attempts)
    result = cast(SqlGeneratorState, graph.invoke(state))
    if result.get("error") is not None:
        logger.error("Agent run finished with an error: %s", result["error"])
    else:
        logger.info("Agent run finished successfully.")
    return result


def _route_after_connection(state: SqlGeneratorState) -> str:
    """Route to the semantic layer, or finish when the DB is unreachable."""
    if state["db_connected"]:
        return "discover_semantic_layer"
    return END


def _route_after_validation(state: SqlGeneratorState) -> str:
    """Route back to generation on failure, to display on success or exhaustion."""
    if state.get("error") is not None:
        return END
    if state.get("validation_error") is not None:
        return "generate_sql"
    return "execute_and_display"
