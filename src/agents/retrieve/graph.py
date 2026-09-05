"""Retrieve agent: returns factual data using the shared SQL generator pipeline.

Runs the shared pipeline and then represents its output as a natural-language
English answer (with a deterministic fallback if the second LLM call fails).
"""

from __future__ import annotations

import logging
from typing import Any

from agents.common.context import AgentContext
from agents.common.llm import LLMError, invoke_llm
from agents.retrieve.prompts import answer_prompt
from agents.sql_generator.graph import build_graph
from agents.sql_generator.graph import run_agent as run_pipeline
from agents.sql_generator.state import SqlGeneratorState

logger = logging.getLogger(__name__)


def run_agent(user_query: str, context: AgentContext) -> SqlGeneratorState:
    """Run the SQL pipeline, then answer the question in plain English."""
    result = run_pipeline(user_query, context)
    rows = result.get("query_rows")
    if result.get("error") is None and rows:
        result["answer"] = _generate_answer(
            context,
            user_query,
            result.get("query_columns") or [],
            rows,
        )
    return result


def _generate_answer(
    context: AgentContext,
    user_query: str,
    columns: list[str],
    rows: list[list[Any]],
) -> str:
    """Ask the LLM to phrase the results as English; fall back on failure."""
    try:
        raw = invoke_llm(context.llm, answer_prompt(user_query, columns, rows)).strip()
    except LLMError:
        logger.warning("Answer generation failed; using a deterministic summary.", exc_info=True)
        return _fallback_answer(user_query, columns, rows)
    return raw if raw else _fallback_answer(user_query, columns, rows)


def _fallback_answer(user_query: str, columns: list[str], rows: list[list[Any]]) -> str:
    """Deterministic English rendering of the result rows."""
    counts = len(rows)
    sentence = (
        f'For "{user_query}" the query returned {counts} row(s).'
        if counts
        else f'For "{user_query}" the query returned no rows.'
    )
    lines = [_format_row(columns, row) for row in rows]
    return sentence if not lines else f"{sentence} " + "; ".join(lines)


def _format_row(columns: list[str], row: list[Any]) -> str:
    pairs = ", ".join(f"{column}: {value}" for column, value in zip(columns, row, strict=False))
    return f"({pairs})"


__all__ = ["build_graph", "run_agent"]
