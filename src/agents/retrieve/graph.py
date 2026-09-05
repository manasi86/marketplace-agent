"""Retrieve agent: returns factual data using the shared SQL generator pipeline."""

from agents.sql_generator.graph import build_graph, run_agent

__all__ = ["build_graph", "run_agent"]
