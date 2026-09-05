"""Prompt templates for the shared SQL generator pipeline."""

from datetime import date
from typing import Any

from agents.sql_generator.dates import oracle_date_range


def intent_prompt(user_query: str) -> str:
    """Build the prompt that parses a natural-language question into intent."""
    return f"""You are a data analyst. Parse the user's question about Oracle database data.

User question:
{user_query}

Respond with ONLY a JSON object in this exact shape:
{{
  "intent": "brief description of what the user wants",
  "entities": {{
    "date_range": "e.g. '2025-01-01 to 2025-12-31' or null",
    "filters": {{
      "column": "value"
    }}
  }},
  "schema_hint": "likely schema or domain name, or null"
}}

Rules:
- intent must be a short actionable phrase, e.g. 'aggregate sales by region'
- entities may be empty; never invent values the user did not mention
- date_range must use ISO format 'YYYY-MM-DD'; use a single date for "today"-style
  requests; represent a range as 'AAAA-MM-DD to BBBB-MM-DD'
- schema_hint should name a schema/domain if the question implies one, else null
"""


def sql_generation_prompt(
    intent: str,
    entities: dict[str, Any],
    semantic_context: str,
    validation_error: str | None = None,
    date_start: date | None = None,
    date_end: date | None = None,
) -> str:
    """Build the prompt that generates an Oracle SELECT statement."""
    parts = [
        "You are an Oracle SQL expert. Generate a single SELECT statement.",
        "",
        f"Intent: {intent}",
        f"Entities: {_render_entities(entities)}",
        "",
        "Available tables/views and their columns:",
        semantic_context,
        "",
        "Constraints:",
        "- Use ONLY tables/views and columns listed above.",
        "- The query must be valid Oracle SQL (observe VARCHAR2, NUMBER, DATE types).",
        "- Start with SELECT or WITH; do not use a trailing semicolon.",
        "- Do not include any explanatory text; output ONLY the SQL.",
    ]
    date_clause = oracle_date_range(date_start, date_end)
    if date_clause is not None:
        parts.append("")
        parts.append(f"Date range: filter by the date using {date_clause}.")
    if validation_error:
        parts.append("")
        parts.append("Your previous attempt FAILED validation with this error:")
        parts.append(validation_error)
        parts.append("Fix the query. Pay close attention to the error; do not repeat the mistake.")
    return "\n".join(parts)


def _render_entities(entities: dict[str, Any]) -> str:
    if not entities:
        return "none"
    return ", ".join(f"{key}={value!r}" for key, value in entities.items())


def sql_generation_system_message() -> str:
    """Return a system-style instruction for SQL generation."""
    return "Generate only valid Oracle SQL SELECT statements. No explanations, no markdown."
