"""Shared SQL generator pipeline (a common agent) built with LangGraph.

Used by the Retrieve agent and by upcoming agents that need to answer
factual data questions. It is intentionally NOT registered in the
coordinator's registry, so the coordinator routes to the agents (e.g.
``retrieve``) that use this pipeline rather than to the pipeline itself.
"""

from agents.sql_generator.graph import build_graph, run_agent
from agents.sql_generator.state import SqlGeneratorState, initial_state

__all__ = [
    "SqlGeneratorState",
    "build_graph",
    "initial_state",
    "run_agent",
]
