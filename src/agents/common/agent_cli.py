"""Shared CLI plumbing for standalone agent entry points.

Every agent's ``cli`` module delegates to :func:`run_agent_cli` so the
bootstrap steps (dotenv, logging, credential checks, context construction)
stay in one place.
"""

import argparse
from collections.abc import Callable
import logging
from typing import Any

from dotenv import load_dotenv

from agents.common.config import get_settings
from agents.common.context import build_context
from agents.common.llm import LLMError
from agents.common.logging_setup import configure_logging

logger = logging.getLogger(__name__)

AgentRunner = Callable[[str, Any], Any]
AgentPrinter = Callable[[Any], None]


def build_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Build an argument parser accepting a query and a ``-v/--verbose`` flag."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "query",
        help="Natural-language question, e.g. 'Show total sales by region for Q1 2026'",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show DEBUG-level logs (prompts, SQL and model responses) on stderr.",
    )
    return parser


def run_agent_cli(
    argv: list[str] | None,
    *,
    prog: str,
    description: str,
    runner: AgentRunner,
    printer: AgentPrinter,
    get_settings_fn: Callable[..., Any] = get_settings,
    build_context_fn: Callable[..., Any] = build_context,
) -> int:
    """Run a standalone agent CLI end to end and return a process exit code.

    ``get_settings_fn`` and ``build_context_fn`` are injectable so agent
    ``cli`` modules can keep module-level references that tests patch.
    """
    load_dotenv()
    args = build_parser(prog, description).parse_args(argv)
    configure_logging(verbose=args.verbose)
    settings = get_settings_fn()
    logger.debug(
        "Agent settings loaded: model=%s max_sql_attempts=%d",
        settings.sql_gen_model,
        settings.max_sql_attempts,
    )

    if not settings.has_llm_credentials:
        logger.error(
            "SQL_GEN_API_KEY is not set. Copy .env.example to .env and fill in your values."
        )
        return 2
    if not settings.has_oracle_credentials:
        logger.error("ORACLE_DSN, ORACLE_USER or ORACLE_PASSWORD missing. Update your .env file.")
        return 2

    try:
        context = build_context_fn(settings)
    except LLMError as exc:
        logger.error("%s", exc)
        return 2

    logger.info("Query: %s", args.query)
    state = runner(args.query, context)
    printer(state)
    if state.get("error") is not None:
        return 1
    return 0
