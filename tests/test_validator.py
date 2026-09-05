"""Tests for SQL validation and sanitisation."""

import pytest

from agents.common.db import DatabaseError, OracleConnection
from agents.sql_generator.validator import is_read_only, sanitize_sql, validate_sql


class _FakeConnection:
    def __init__(self, explain_error: str | None = None) -> None:
        self._explain_error = explain_error
        self.explained: list[str] = []

    def explain_query(self, sql: str) -> None:
        self.explained.append(sql)
        if self._explain_error is not None:
            raise DatabaseError(self._explain_error)


def test_validate_sql_rejects_empty() -> None:
    connection = _FakeConnection()
    valid, error = validate_sql("   -- only a comment\n", connection)  # type: ignore[arg-type]
    assert valid is False
    assert error == "Generated SQL is empty."


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET x = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "CREATE TABLE t (x NUMBER)",
        "ALTER TABLE t ADD y NUMBER",
        "TRUNCATE TABLE t",
        "GRANT SELECT ON t TO x",
        "REVOKE SELECT ON t FROM x",
        "MERGE INTO t USING s ON (a=b) WHEN MATCHED THEN UPDATE SET c=1",
        "CALL pkg.proc()",
        "EXEC pkg.proc",
        "EXECUTE IMMEDIATE 'x'",
        "BEGIN NULL; END;",
        "COMMIT",
    ],
)
def test_validate_sql_rejects_non_read_only(sql: str) -> None:
    connection = _FakeConnection()
    valid, error = validate_sql(sql, connection)  # type: ignore[arg-type]
    assert valid is False
    assert error is not None


def test_validate_sql_select_only_message() -> None:
    connection = _FakeConnection()
    _, error = validate_sql("INSERT INTO t VALUES (1)", connection)  # type: ignore[arg-type]
    assert error == "Only SELECT statements are allowed."


def test_validate_sql_forbidden_keyword_message() -> None:
    connection = _FakeConnection()
    _, error = validate_sql("SELECT * FROM dual; DELETE FROM t", connection)  # type: ignore[arg-type]
    assert error == "Forbidden keyword detected: DELETE."


def test_validate_sql_comment_stripped_does_not_trip_guard() -> None:
    connection = _FakeConnection()
    valid, error = validate_sql("SELECT 1 FROM DUAL -- DROP TABLE x", connection)  # type: ignore[arg-type]
    assert valid is True
    assert error is None
    assert connection.explained == ["SELECT 1 FROM DUAL -- DROP TABLE x"]


def test_validate_sql_valid_select_passes() -> None:
    sql = "SELECT region, SUM(total) FROM SALES.VW_SALES_SUMMARY GROUP BY region"
    connection = _FakeConnection()
    valid, error = validate_sql(sql, connection)  # type: ignore[arg-type]
    assert valid is True
    assert error is None
    assert connection.explained == [sql]


def test_validate_sql_propagates_explain_error() -> None:
    connection = _FakeConnection(explain_error="ORA-00904: invalid identifier")
    valid, error = validate_sql("SELECT bad_col FROM SALES.VW_X", connection)  # type: ignore[arg-type]
    assert valid is False
    assert "ORA-00904" in (error or "")


def test_is_read_only() -> None:
    assert is_read_only("SELECT * FROM dual")
    assert is_read_only("WITH x AS (SELECT 1 FROM dual) SELECT * FROM x")
    assert not is_read_only("")
    assert not is_read_only("DROP TABLE x")
    assert not is_read_only("SELECT * FROM dual; UPDATE t SET a=1")


def test_sanitize_sql_removes_fences_and_comments() -> None:
    raw = """
```sql
SELECT 1 + 1 AS result -- inline note
  FROM dual /* block
  comment */
;
```
"""
    cleaned = sanitize_sql(raw)
    assert cleaned == "SELECT 1 + 1 AS result\nFROM dual"
    assert cleaned.endswith(";") is False


def test_sanitize_sql_strips_trailing_semicolon_without_fences() -> None:
    assert sanitize_sql("SELECT * FROM dual;") == "SELECT * FROM dual"


def test_sanitize_sql_fenced_without_closing_fence() -> None:
    assert sanitize_sql("```sql\nSELECT 1 FROM dual") == "SELECT 1 FROM dual"


def test_sanitize_sql_preserves_plain_statement() -> None:
    assert sanitize_sql("SELECT currency FROM t") == "SELECT currency FROM t"


def test_sanitize_sql_empty_stays_empty() -> None:
    assert sanitize_sql(" ") == ""


def test_oracle_connection_import_is_available() -> None:
    assert OracleConnection is not None
