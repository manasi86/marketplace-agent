"""Graph node implementations for the shared SQL generator pipeline."""

from collections.abc import Callable
import logging
from time import perf_counter
from typing import Any

from agents.common.context import AgentContext
from agents.common.db import DatabaseError
from agents.common.llm import LLMError, invoke_llm
from agents.sql_generator.prompts import sql_generation_prompt
from agents.sql_generator.state import SqlGeneratorState
from agents.sql_generator.validator import sanitize_sql, validate_sql

logger = logging.getLogger(__name__)

NodeFn = Callable[[SqlGeneratorState], dict[str, Any]]


def build_nodes(context: AgentContext) -> dict[str, NodeFn]:
    """Create the agent's graph nodes closed over the shared agent context."""

    def check_db_connection(state: SqlGeneratorState) -> dict[str, Any]:
        del state
        logger.info("Checking database connection...")
        connected = context.connection.check_connection()
        if not connected:
            logger.error("Database connection check failed.")
            return {
                "db_connected": False,
                "error": "Could not connect to the Oracle database.",
                "done": True,
                "logs": [("check_db_connection", "Database connection check: FAILED")],
            }
        logger.info("Database connection check: OK")
        return {
            "db_connected": True,
            "logs": [("check_db_connection", "Database connection check: OK")],
        }

    def discover_semantic_layer(state: SqlGeneratorState) -> dict[str, Any]:
        logger.info(
            "Discovering semantic layer%s...",
            f" (schema hint: {state['schema_hint']})" if state.get("schema_hint") else "",
        )
        metadata = context.semantic.discover(context.connection)
        formatted = context.semantic.format_for_prompt(state.get("schema_hint"))
        schemas = list(metadata)
        covered = f" (schema hint: {state['schema_hint']})" if state.get("schema_hint") else ""
        logger.info(
            "Discovered %d schema(s): %s%s",
            len(schemas),
            ", ".join(sorted(schemas)),
            covered,
        )
        return {
            "semantic_metadata": metadata,
            "semantic_context": formatted,
            "logs": [
                (
                    "discover_semantic_layer",
                    f"Discovered semantic layer: {len(schemas)} schema(s){covered}",
                ),
            ],
        }

    def generate_sql(state: SqlGeneratorState) -> dict[str, Any]:
        attempt_note = (
            " (fix attempt after validation error)" if state.get("validation_error") else ""
        )
        logger.info("Generating SQL%s...", attempt_note)
        prompt = sql_generation_prompt(
            state["user_query"],
            state["semantic_context"],
            state.get("validation_error"),
        )
        logger.debug("SQL generation prompt:\n%s", prompt)
        try:
            raw = invoke_llm(context.llm, prompt)
        except LLMError as exc:
            logger.error("SQL generation failed: %s", exc)
            return {
                "error": str(exc),
                "done": True,
                "logs": [("generate_sql", f"SQL generation failed: {exc}")],
            }
        sql = sanitize_sql(raw)
        logger.debug("Raw model response:\n%s", raw)
        logger.info("Generated SQL%s: %s", attempt_note, sql)
        return {"sql_query": sql, "logs": [("generate_sql", f"Generated SQL{attempt_note}")]}

    def validate_sql_query(state: SqlGeneratorState) -> dict[str, Any]:
        sql = state["sql_query"]
        logger.info("Validating SQL (SELECT-only + EXPLAIN)...")
        valid, error_message = validate_sql(sql, context.connection)
        if valid:
            logger.info("SQL validation: PASSED")
            return {
                "validation_error": None,
                "logs": [("validate_sql", "SQL validation: PASSED (SELECT-only + EXPLAIN OK)")],
            }
        attempts = state["attempt_count"] + 1
        logger.warning(
            "SQL validation FAILED (attempt %d/%d): %s",
            attempts,
            state["max_attempts"],
            error_message,
        )
        result: dict[str, Any] = {
            "attempt_count": attempts,
            "validation_error": error_message,
            "logs": [
                (
                    "validate_sql",
                    (
                        f"SQL validation FAILED "
                        f"(attempt {attempts}/{state['max_attempts']}): {error_message}"
                    ),
                )
            ],
        }
        if attempts >= state["max_attempts"]:
            result["error"] = (
                f"SQL validation failed after {state['max_attempts']} attempts. "
                f"Last error: {error_message}"
            )
            result["done"] = True
        return result

    def execute_and_display(state: SqlGeneratorState) -> dict[str, Any]:
        logger.info("Executing SQL query...")
        start = perf_counter()
        try:
            query_result = context.connection.execute_query(state["sql_query"])
        except DatabaseError as exc:
            logger.error("Query execution failed: %s", exc)
            return {
                "error": str(exc),
                "done": True,
                "logs": [("execute_and_display", f"Query execution failed: {exc}")],
            }
        elapsed_ms = (perf_counter() - start) * 1000.0
        logger.info(
            "Query executed successfully: %d row(s) in %.1f ms",
            len(query_result.rows),
            elapsed_ms,
        )
        return {
            "query_columns": query_result.columns,
            "query_rows": query_result.rows,
            "execution_time_ms": elapsed_ms,
            "done": True,
            "logs": [
                (
                    "execute_and_display",
                    (
                        f"Query executed successfully: {len(query_result.rows)} row(s) "
                        f"in {elapsed_ms:.1f} ms"
                    ),
                ),
            ],
        }

    return {
        "check_db_connection": check_db_connection,
        "discover_semantic_layer": discover_semantic_layer,
        "generate_sql": generate_sql,
        "validate_sql": validate_sql_query,
        "execute_and_display": execute_and_display,
    }
