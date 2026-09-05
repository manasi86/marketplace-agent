"""Validation of generated SQL: read-only guard plus compilation check."""

import re

from lib.sql_generator.db import DatabaseError, OracleConnection

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(?:"
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|MERGE|"
    r"CALL|EXEC|EXECUTE|BEGIN|DECLARE|COMMENT|COMMIT|ROLLBACK|FLASHBACK|"
    r"PURGE|DBMS_|UTL_|SET\s+TRANSACTION"
    r")\b",
    re.IGNORECASE,
)

_READ_ONLY_START = re.compile(r"^\s*\(*\s*(?:SELECT|WITH)\b", re.IGNORECASE)


def validate_sql(sql: str, connection: OracleConnection) -> tuple[bool, str | None]:
    """Validate generated SQL before execution.

    Performs two checks:

    1. The statement must be a read-only SELECT (or a WITH/CTE that resolves to
       a SELECT) and must not contain any mutating or administrative keywords.
    2. The statement must compile, verified via ``EXPLAIN PLAN`` on the live
       Oracle connection (catches unknown tables, bad columns and syntax
       errors surfaced by the parser).

    Returns ``(valid, error_message)`` where ``error_message`` is ``None`` when
    the statement passes every check.
    """
    clean = _strip_comments(sql)
    normalized = clean.strip()
    if not normalized:
        return False, "Generated SQL is empty."
    if not _READ_ONLY_START.match(normalized):
        return False, "Only SELECT statements are allowed."
    if _FORBIDDEN_KEYWORDS.search(normalized):
        match = _FORBIDDEN_KEYWORDS.search(normalized)
        assert match is not None
        return False, f"Forbidden keyword detected: {match.group(0).split()[0].upper()}."
    try:
        connection.explain_query(sql)
    except DatabaseError as exc:
        return False, str(exc)
    return True, None


def is_read_only(sql: str) -> bool:
    """Return True when the statement is a read-only SELECT or WITH query."""
    normalized = _strip_comments(sql).strip()
    return bool(
        normalized
        and _READ_ONLY_START.match(normalized)
        and not _FORBIDDEN_KEYWORDS.search(normalized)
    )


def sanitize_sql(raw: str) -> str:
    """Normalise a model response into a single executable SQL statement."""
    cleaned = _strip_fences(_strip_comments(raw))
    lines: list[str] = []
    for line in cleaned.splitlines():
        processed = line.rstrip(";").strip()
        if processed:
            lines.append(processed)
    return "\n".join(lines).strip()


def _strip_comments(sql: str) -> str:
    """Remove SQL block and line comments from a statement."""
    without_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    without_line = re.sub(r"--.*$", " ", without_block, flags=re.MULTILINE)
    return without_line


def _strip_fences(raw: str) -> str:
    """Remove surrounding markdown code fences if present."""
    lines = raw.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
