"""Standalone CLI entry point for the SQL generator agent."""

import argparse
import sys

from dotenv import load_dotenv

from lib.sql_generator.config import get_settings
from lib.sql_generator.context import build_context
from lib.sql_generator.display import print_agent_output
from lib.sql_generator.graph import run_agent
from lib.sql_generator.llm import LLMError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="sql-generator",
        description="Generate and run Oracle SQL from a natural-language question.",
    )
    parser.add_argument(
        "query",
        help="Natural-language question, e.g. 'Show total sales by region for Q1 2026'",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the agent and render results; return a process exit code."""
    load_dotenv()
    args = parse_args(argv)
    settings = get_settings()

    if not settings.has_llm_credentials:
        print(
            "ERROR: SQL_GEN_API_KEY is not set. Copy .env.example to .env "
            "and fill in your values.",
            file=sys.stderr,
        )
        return 2
    if not settings.has_oracle_credentials:
        print(
            "ERROR: ORACLE_DSN, ORACLE_USER or ORACLE_PASSWORD missing. Update your .env file.",
            file=sys.stderr,
        )
        return 2

    try:
        context = build_context(settings)
    except LLMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    state = run_agent(args.query, context)
    print_agent_output(state)
    if state.get("error") is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
