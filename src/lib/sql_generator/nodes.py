"""Graph node implementations for the SQL generator agent."""

from collections.abc import Callable
import json
import re
from time import perf_counter
from typing import Any, cast

from lib.sql_generator.context import AgentContext
from lib.sql_generator.dates import parse_date_range
from lib.sql_generator.db import DatabaseError
from lib.sql_generator.llm import LLMError, invoke_llm
from lib.sql_generator.prompts import intent_prompt, sql_generation_prompt
from lib.sql_generator.state import AgentState
from lib.sql_generator.validator import sanitize_sql, validate_sql

_NodeFn = Callable[[AgentState], dict[str, Any]]


def build_nodes(context: AgentContext) -> dict[str, _NodeFn]:
    """Create the six graph nodes closed over the shared agent context."""

    def understand_intent(state: AgentState) -> dict[str, Any]:
        try:
            raw = invoke_llm(context.llm, intent_prompt(state["user_query"]))
            parsed = _parse_intent_json(raw)
        except LLMError:
            parsed = {
                "intent": state["user_query"],
                "entities": {},
                "schema_hint": None,
            }
        schema_hint = parsed.get("schema_hint")
        entities = parsed.get("entities") or {}
        date_start, date_end = parse_date_range(entities)
        return {
            "intent": str(parsed.get("intent") or state["user_query"]),
            "entities": entities,
            "schema_hint": (str(schema_hint) if schema_hint else None),
            "date_start": date_start,
            "date_end": date_end,
            "logs": [
                (
                    "understand_intent",
                    f"Understood intent: {parsed.get('intent') or state['user_query']}",
                ),
            ],
        }

    def check_db_connection(state: AgentState) -> dict[str, Any]:
        connected = context.connection.check_connection()
        if not connected:
            return {
                "db_connected": False,
                "error": "Could not connect to the Oracle database.",
                "done": True,
                "logs": [("check_db_connection", "Database connection check: FAILED")],
            }
        return {
            "db_connected": True,
            "logs": [("check_db_connection", "Database connection check: OK")],
        }

    def discover_semantic_layer(state: AgentState) -> dict[str, Any]:
        metadata = context.semantic.discover(context.connection)
        formatted = context.semantic.format_for_prompt(state.get("schema_hint"))
        schemas = list(metadata)
        covered = f" (schema hint: {state['schema_hint']})" if state.get("schema_hint") else ""
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

    def generate_sql(state: AgentState) -> dict[str, Any]:
        prompt = sql_generation_prompt(
            state["intent"],
            state["entities"],
            state["semantic_context"],
            state.get("validation_error"),
            date_start=state.get("date_start"),
            date_end=state.get("date_end"),
        )
        try:
            raw = invoke_llm(context.llm, prompt)
        except LLMError as exc:
            return {
                "error": str(exc),
                "done": True,
                "logs": [("generate_sql", f"SQL generation failed: {exc}")],
            }
        sql = sanitize_sql(raw)
        attempt_note = (
            " (fix attempt after validation error)" if state.get("validation_error") else ""
        )
        return {"sql_query": sql, "logs": [("generate_sql", f"Generated SQL{attempt_note}")]}

    def validate_sql_query(state: AgentState) -> dict[str, Any]:
        sql = state["sql_query"]
        valid, error_message = validate_sql(sql, context.connection)
        if valid:
            return {
                "validation_error": None,
                "logs": [("validate_sql", "SQL validation: PASSED (SELECT-only + EXPLAIN OK)")],
            }
        attempts = state["attempt_count"] + 1
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

    def execute_and_display(state: AgentState) -> dict[str, Any]:
        start = perf_counter()
        try:
            query_result = context.connection.execute_query(state["sql_query"])
        except DatabaseError as exc:
            return {
                "error": str(exc),
                "done": True,
                "logs": [("execute_and_display", f"Query execution failed: {exc}")],
            }
        elapsed_ms = (perf_counter() - start) * 1000.0
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
        "understand_intent": understand_intent,
        "check_db_connection": check_db_connection,
        "discover_semantic_layer": discover_semantic_layer,
        "generate_sql": generate_sql,
        "validate_sql": validate_sql_query,
        "execute_and_display": execute_and_display,
    }


def _parse_intent_json(raw: str) -> dict[str, Any]:
    """Tolerantly parse a JSON object from a model response."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return {"intent": "", "entities": {}, "schema_hint": None}
    try:
        return cast("dict[str, Any]", json.loads(match.group(0)))
    except json.JSONDecodeError:
        return {"intent": "", "entities": {}, "schema_hint": None}
