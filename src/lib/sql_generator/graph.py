"""LangGraph wiring for the SQL generator agent."""

from collections.abc import Callable
import functools
import logging
from time import perf_counter
from typing import Any, cast

from langgraph.graph import END, StateGraph

from lib.sql_generator.context import AgentContext
from lib.sql_generator.nodes import NodeFn, build_nodes
from lib.sql_generator.observability import observe_run, observe_step
from lib.sql_generator.state import AgentState, initial_state

logger = logging.getLogger(__name__)

NODE_NAMES = (
    "understand_intent",
    "check_db_connection",
    "discover_semantic_layer",
    "generate_sql",
    "validate_sql",
    "execute_and_display",
)


def _log_step(name: str) -> Callable[[NodeFn], NodeFn]:
    """Return a decorator that logs each graph step with its duration."""

    def _decorator(func: NodeFn) -> NodeFn:
        @functools.wraps(func)
        def _wrapped(state: AgentState) -> dict[str, Any]:
            logger.info("Step [%s] started", name)
            start = perf_counter()
            try:
                result = func(state)
            except Exception:
                logger.exception("Step [%s] failed", name)
                raise
            elapsed_ms = (perf_counter() - start) * 1000.0
            logger.info("Step [%s] completed in %.1f ms", name, elapsed_ms)
            return result

        return _wrapped

    return _decorator


def build_graph(context: AgentContext) -> Any:
    """Compile the agent's StateGraph with traced node functions."""
    nodes = build_nodes(context)
    graph = StateGraph(AgentState)
    for name in NODE_NAMES:
        graph.add_node(  # type: ignore[call-overload]
            name, observe_step(name)(_log_step(name)(nodes[name]))
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
def run_agent(user_query: str, context: AgentContext) -> AgentState:
    """Execute the full agent pipeline for a natural-language question."""
    logger.info("Starting agent run for query: %s", user_query)
    graph = build_graph(context)
    state = initial_state(user_query, context.settings.max_sql_attempts)
    result = cast(AgentState, graph.invoke(state))
    if result.get("error") is not None:
        logger.error("Agent run finished with an error: %s", result["error"])
    else:
        logger.info("Agent run finished successfully.")
    return result


def _route_after_connection(state: AgentState) -> str:
    """Route to the semantic layer, or finish when the DB is unreachable."""
    if state["db_connected"]:
        return "discover_semantic_layer"
    return END


def _route_after_validation(state: AgentState) -> str:
    """Route back to generation on failure, to display on success or exhaustion."""
    if state.get("error") is not None:
        return END
    if state.get("validation_error") is not None:
        return "generate_sql"
    return "execute_and_display"
