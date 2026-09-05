"""Prompt templates for the Retrieve agent."""

from typing import Any


def answer_prompt(user_query: str, columns: list[str], rows: list[list[Any]]) -> str:
    """Build the prompt that turns query results into a fluent English answer."""
    header = " | ".join(str(column) for column in columns)
    table_lines = "\n".join(
        f"{index}. " + " | ".join(str(cell) for cell in row)
        for index, row in enumerate(rows, start=1)
    )
    return f"""You are a helpful data assistant. Answer the user's question in plain English.

Use ONLY the data in the query results below. Be concise and factual, cite exact
numbers where possible, and never mention SQL, tables or columns. If a "today"
comparison is asked but the data has no current-date row, state that clearly.

User question:
{user_query}

Query results:
{header if header else "(no columns)"}
{table_lines if table_lines else "(no rows)"}
"""
