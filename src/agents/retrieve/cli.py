"""Standalone CLI entry point for the Retrieve agent."""

import argparse

from agents.common.agent_cli import build_parser, run_agent_cli
from agents.common.config import get_settings
from agents.common.context import build_context
from agents.retrieve.display import print_agent_output
from agents.retrieve.graph import run_agent

_DESCRIPTION = (
    "Return factual data for a natural-language question by running the "
    "shared SQL generator pipeline."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser("retrieve-agent", _DESCRIPTION).parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Retrieve agent and render results; return a process exit code."""
    return run_agent_cli(
        argv,
        prog="retrieve-agent",
        description=_DESCRIPTION,
        runner=run_agent,
        printer=print_agent_output,
        get_settings_fn=get_settings,
        build_context_fn=build_context,
    )


if __name__ == "__main__":
    raise SystemExit(main())
