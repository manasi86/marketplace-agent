"""Standalone CLI entry point for the coordinator agent."""

import argparse
import logging
from typing import Any, cast

from dotenv import load_dotenv

from agents.common.config import get_settings
from agents.common.context import build_context
from agents.common.display import Results, print_agent_output
from agents.common.llm import LLMError
from agents.common.logging_setup import configure_logging
from agents.common.state import BaseAgentState
from agents.coordinator.graph import run_agent

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="agents coord",
        description="Classify a user request and route it to the matching agent.",
    )
    parser.add_argument(
        "query",
        help="Natural-language request, e.g. 'Show total sales by region for Q1 2026'",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show DEBUG-level logs (prompts, SQL and model responses) on stderr.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the coordinator and render results; return a process exit code."""
    load_dotenv()
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    settings = get_settings()
    logger.debug("Agent settings loaded: model=%s", settings.sql_gen_model)

    if not settings.has_llm_credentials:
        logger.error(
            "SQL_GEN_API_KEY is not set. Copy .env.example to .env and fill in your values."
        )
        return 2

    try:
        context = build_context(settings)
    except LLMError as exc:
        logger.error("%s", exc)
        return 2

    logger.info("Query: %s", args.query)
    state = run_agent(args.query, context)
    print_agent_output(
        cast(BaseAgentState, state),
        results=_build_results(state),
        failure_message=state.get("error"),
        answer=state.get("answer"),
    )
    if state.get("error") is not None:
        return 1
    return 0


def _build_results(state: dict[str, Any]) -> Results | None:
    """Extract an optional tabular result from the merged coordinator state."""
    rows = state.get("query_rows")
    if rows is None:
        return None
    columns = state.get("query_columns")
    elapsed = state.get("execution_time_ms")
    sql = state.get("sql_query")
    attempts = state.get("attempt_count")
    return Results(
        columns=list(columns) if isinstance(columns, list) else [],
        rows=list(rows) if isinstance(rows, list) else [],
        execution_time_ms=float(elapsed) if isinstance(elapsed, (int, float)) else None,
        sql_query=str(sql or ""),
        attempt_count=int(attempts or 0),
    )


if __name__ == "__main__":
    raise SystemExit(main())
