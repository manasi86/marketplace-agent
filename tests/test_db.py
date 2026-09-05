"""Tests for the Oracle database layer."""

from typing import Any

import oracledb
import pytest

from lib.sql_generator.config import Settings
from lib.sql_generator.db import SYSTEM_SCHEMAS, DatabaseError, OracleConnection, _connect_oracle


class _FakeCursor:
    """Minimal stand-in for an oracledb cursor."""

    def __init__(
        self,
        description: list[tuple[str, ...]] | None = None,
        rows: list[tuple[Any, ...]] | None = None,
        execute_error: Exception | None = None,
    ) -> None:
        self.description = description
        self._rows = rows if rows is not None else []
        self._execute_error = execute_error
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = 0

    def execute(self, sql: str, *args: Any) -> None:
        self.executed.append((sql, args))
        if self._execute_error is not None:
            raise self._execute_error

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def close(self) -> None:
        self.closed += 1

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return exc[0] is None


class _FakeConnection:
    """Minimal stand-in for an oracledb connection."""

    def __init__(self, cursors: list[_FakeCursor] | None = None) -> None:
        self._cursors = list(cursors or [])
        self.closed = 0

    def cursor(self) -> _FakeCursor:
        if self._cursors:
            return self._cursors.pop(0)
        return _FakeCursor(
            description=[("COL",)],
            rows=[("value",)],
        )

    def close(self) -> None:
        self.closed += 1


def _settings(**overrides: str | int | bool) -> Settings:
    defaults: dict[str, str | int | bool] = {
        "sql_gen_api_key": "key",
        "oracle_dsn": "db:1521/orcl",
        "oracle_user": "scott",
        "oracle_password": "tiger",
        "langfuse_enabled": False,
    }
    defaults.update(overrides)
    return Settings.model_validate(defaults)


def test_connectoracle_importable() -> None:
    assert callable(_connect_oracle)


def test_connect_missing_credentials_raises() -> None:
    connection = OracleConnection(_settings(oracle_dsn="", oracle_user="", oracle_password=""))
    with pytest.raises(DatabaseError, match="Oracle credentials are missing"):
        connection.connect()


def test_connect_open_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise oracledb.Error("ORACLE: unable to connect")

    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", fail)
    connection = OracleConnection(_settings())
    with pytest.raises(DatabaseError, match="Failed to connect"):
        connection.connect()


def test_connect_and_reconnect_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeConnection()
    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", lambda *a, **k: fake)
    connection = OracleConnection(_settings())
    connection.connect()
    connection.connect()
    assert connection._connection is fake  # type: ignore


def test_check_connection_true(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor()
    fake = _FakeConnection([cursor])
    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", lambda *a, **k: fake)
    connection = OracleConnection(_settings())
    assert connection.check_connection() is True
    assert cursor.executed == [("SELECT 1 FROM DUAL", ())]


def test_check_connection_connect_failure_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise oracledb.Error("nope")

    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", fail)
    connection = OracleConnection(_settings())
    assert connection.check_connection() is False


def test_check_connection_execute_failure_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor(execute_error=oracledb.Error("boom"))
    fake = _FakeConnection([cursor])
    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", lambda *a, **k: fake)
    connection = OracleConnection(_settings())
    assert connection.check_connection() is False


def test_execute_query_returns_columns_and_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor(
        description=[("REGION",), ("TOTAL",)],
        rows=[("East", 100.0), ("West", 50.0)],
    )
    fake = _FakeConnection([cursor])
    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", lambda *a, **k: fake)
    connection = OracleConnection(_settings())
    result = connection.execute_query("SELECT region, total FROM vw_sales")
    assert result.columns == ["REGION", "TOTAL"]
    assert result.rows == [["East", 100.0], ["West", 50.0]]


def test_execute_query_without_description(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor(description=None)
    fake = _FakeConnection([cursor])
    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", lambda *a, **k: fake)
    connection = OracleConnection(_settings())
    result = connection.execute_query("BEGIN NULL; END;")
    assert result.columns == []
    assert result.rows == []


def test_execute_query_oracle_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor(execute_error=oracledb.Error("ORA-00904"))
    fake = _FakeConnection([cursor])
    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", lambda *a, **k: fake)
    connection = OracleConnection(_settings())
    with pytest.raises(DatabaseError, match="Query execution failed"):
        connection.execute_query("SELECT bad FROM x")


def test_explain_query_success(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor()
    fake = _FakeConnection([cursor])
    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", lambda *a, **k: fake)
    connection = OracleConnection(_settings())
    connection.explain_query("SELECT 1 FROM dual")
    assert cursor.executed == [("EXPLAIN PLAN FOR SELECT 1 FROM dual", ())]


def test_explain_query_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor(execute_error=oracledb.Error("ORA-00942"))
    fake = _FakeConnection([cursor])
    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", lambda *a, **k: fake)
    connection = OracleConnection(_settings())
    with pytest.raises(DatabaseError, match="ORA-00942"):
        connection.explain_query("SELECT * FROM nope")


def test_fetch_schema_builds_nested_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    owners_cursor = _FakeCursor(
        description=[("OWNER",)],
        rows=[("SALES",), ("INVENTORY",)],
    )
    sales_cursor = _FakeCursor(
        description=[
            ("OWNER",),
            ("TABLE_NAME",),
            ("TABLE_TYPE",),
            ("TABLE_COMMENTS",),
            ("COLUMN_NAME",),
            ("DATA_TYPE",),
            ("COLUMN_COMMENTS",),
        ],
        rows=[
            ("SALES", "VW_SALES_SUMMARY", "VIEW", "Sales summary", "REGION", "VARCHAR2", "Region"),
            ("SALES", "VW_SALES_SUMMARY", "VIEW", "Sales summary", "TOTAL", "NUMBER", None),
            ("SALES", "T_ORDERS", "TABLE", None, "ORDER_ID", "NUMBER", "Order PK"),
        ],
    )
    inventory_cursor = _FakeCursor(
        description=[
            ("OWNER",),
            ("TABLE_NAME",),
            ("TABLE_TYPE",),
            ("TABLE_COMMENTS",),
            ("COLUMN_NAME",),
            ("DATA_TYPE",),
            ("COLUMN_COMMENTS",),
        ],
        rows=[],
    )
    fake = _FakeConnection([owners_cursor, inventory_cursor, sales_cursor])
    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", lambda *a, **k: fake)
    connection = OracleConnection(_settings())
    schema = connection.fetch_schema()

    assert set(schema) == {"SALES", "INVENTORY"}
    sales = schema["SALES"]["tables"]["VW_SALES_SUMMARY"]
    assert sales["type"] == "VIEW"
    assert sales["description"] == "Sales summary"
    assert sales["columns"]["REGION"]["type"] == "VARCHAR2"
    assert sales["columns"]["TOTAL"]["description"] is None
    orders = schema["SALES"]["tables"]["T_ORDERS"]
    assert orders["description"] is None
    assert schema["INVENTORY"]["tables"] == {}

    owners_sql, owners_args = owners_cursor.executed[0]
    assert "OWNER NOT IN (" in owners_sql
    assert owners_sql.endswith(")")
    assert owners_args == (SYSTEM_SCHEMAS,)
    assert ":1" in owners_sql
    assert ":0" not in owners_sql
    sales_sql, sales_args = sales_cursor.executed[0]
    assert "SELECT c.OWNER" in sales_sql
    assert sales_args == ({"owner": "SALES"},)


def test_fetch_schema_skips_falsy_owners(monkeypatch: pytest.MonkeyPatch) -> None:
    owners_cursor = _FakeCursor(description=[("OWNER",)], rows=[("",), (None,)])
    fake = _FakeConnection([owners_cursor])
    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", lambda *a, **k: fake)
    connection = OracleConnection(_settings())
    assert connection.fetch_schema() == {}


def test_close_sets_connection_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeConnection()
    monkeypatch.setattr("lib.sql_generator.db._connect_oracle", lambda *a, **k: fake)
    connection = OracleConnection(_settings())
    connection.connect()
    connection.close()
    assert connection._connection is None
    assert fake.closed == 1


def test_connect_oracle_function(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeConnection()
    monkeypatch.setattr(
        "lib.sql_generator.db._oracle_connect",
        lambda user, password, dsn: fake,
    )
    connection = _connect_oracle("u", "p", "d")
    assert connection is fake  # type: ignore


def test_close_without_connection_is_noop() -> None:
    connection = OracleConnection(_settings())
    connection.close()
    assert connection._connection is None


def test_query_dicts_without_description(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = _FakeCursor(description=None)
    fake = _FakeConnection([cursor])
    monkeypatch.setattr("lib.sql_generator.db._oracle_connect", lambda *a, **k: fake)
    connection = OracleConnection(_settings())
    connection.connect()
    assert connection._query_dicts("SELECT 1 FROM dual") == []


def test_operations_without_active_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lib.sql_generator.db._connect_oracle",
        lambda *a, **k: _FakeConnection(),
    )
    connection = OracleConnection(_settings())
    connection._connection = None
    with pytest.raises(DatabaseError, match="No active connection"):
        connection._execute("SELECT 1 FROM DUAL")
    with pytest.raises(DatabaseError, match="No active connection"):
        connection._query_dicts("SELECT 1 FROM DUAL")
