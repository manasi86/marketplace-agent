"""Standalone CLI entry point for the shared SQL generator pipeline."""

import argparse

from agents.common.agent_cli import build_parser, run_agent_cli
from agents.common.config import get_settings
from agents.common.context import build_context
from agents.sql_generator.display import print_agent_output
from agents.sql_generator.graph import run_agent

_DESCRIPTION = "Generate and run Oracle SQL from a natural-language question."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser("sql-generator", _DESCRIPTION).parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and render results; return a process exit code."""
    return run_agent_cli(
        argv,
        prog="sql-generator",
        description=_DESCRIPTION,
        runner=run_agent,
        printer=print_agent_output,
        get_settings_fn=get_settings,
        build_context_fn=build_context,
    )


if __name__ == "__main__":
    raise SystemExit(main())
