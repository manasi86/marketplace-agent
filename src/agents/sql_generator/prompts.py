"""Prompt templates for the shared SQL generator pipeline."""


def sql_generation_prompt(
    user_query: str,
    semantic_context: str,
    validation_error: str | None = None,
) -> str:
    """Build the prompt that generates an Oracle SELECT statement."""
    parts = [
        "You are an Oracle SQL expert. Generate a single SELECT statement.",
        "",
        f"Question: {user_query}",
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
    if validation_error:
        parts.append("")
        parts.append("Your previous attempt FAILED validation with this error:")
        parts.append(validation_error)
        parts.append("Fix the query. Pay close attention to the error; do not repeat the mistake.")
    return "\n".join(parts)


def sql_generation_system_message() -> str:
    """Return a system-style instruction for SQL generation."""
    return "Generate only valid Oracle SQL SELECT statements. No explanations, no markdown."
