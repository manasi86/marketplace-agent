"""LangGraph wiring for the SQL generator agent."""

from typing import Any, cast

from langgraph.graph import END, StateGraph

from lib.sql_generator.context import AgentContext
from lib.sql_generator.nodes import build_nodes
from lib.sql_generator.observability import observe_run, observe_step
from lib.sql_generator.state import AgentState, initial_state

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
    graph = StateGraph(AgentState)
    for name in NODE_NAMES:
        graph.add_node(  # type: ignore[call-overload]
            name, observe_step(name)(nodes[name])
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
    graph = build_graph(context)
    state = initial_state(user_query, context.settings.max_sql_attempts)
    return cast(AgentState, graph.invoke(state))


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
