"""Oracle database access using oracledb in thin mode, shared by agents."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from typing import Any

import oracledb

from agents.common.config import Settings

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Raised when any database operation fails."""


_oracle_connect = oracledb.connect


def _connect_oracle(user: str, password: str, dsn: str) -> oracledb.Connection:
    """Open a new oracledb connection; extracted for testability."""
    return _oracle_connect(user=user, password=password, dsn=dsn)


@dataclass(frozen=True)
class QueryResult:
    """Columns and rows returned by a successful SELECT statement."""

    columns: list[str]
    rows: list[list[Any]]


class OracleConnection:
    """Thin wrapper around a single oracledb connection."""

    def __init__(self, settings: Settings) -> None:
        """Store settings; the connection is created lazily."""
        self._settings = settings
        self._connection: oracledb.Connection | None = None

    def connect(self) -> None:
        """Open the connection if it is not already open."""
        if self._connection is not None:
            return
        if not self._settings.has_oracle_credentials:
            raise DatabaseError(
                "Oracle credentials are missing. Set ORACLE_DSN, ORACLE_USER "
                "and ORACLE_PASSWORD in the .env file."
            )
        try:
            logger.info("Connecting to Oracle database...")
            self._connection = _connect_oracle(
                user=self._settings.oracle_user,
                password=self._settings.oracle_password,
                dsn=self._settings.oracle_dsn,
            )
            logger.info("Connected to Oracle (dsn=%s)", self._settings.oracle_dsn)
        except oracledb.Error as exc:
            logger.error("Failed to connect to Oracle: %s", exc)
            raise DatabaseError(f"Failed to connect to Oracle: {exc}") from exc

    def check_connection(self) -> bool:
        """Return True when the database responds to a trivial query."""
        try:
            self.connect()
        except DatabaseError:
            return False
        try:
            self._execute("SELECT 1 FROM DUAL")
        except DatabaseError:
            return False
        return True

    def execute_query(self, sql: str) -> QueryResult:
        """Execute a SELECT statement and return its columns and rows."""
        self.connect()
        logger.info("Executing query (%d chars)", len(sql))
        logger.debug("SQL:\n%s", sql)
        try:
            connection = self._connection
            assert connection is not None
            with connection.cursor() as cursor:
                cursor.execute(sql)
                if cursor.description is None:
                    return QueryResult(columns=[], rows=[])
                columns = [description[0] for description in cursor.description]
                rows = [list(row) for row in cursor.fetchall()]
                logger.info("Query complete: %d row(s)", len(rows))
                return QueryResult(columns=columns, rows=rows)
        except oracledb.Error as exc:
            logger.error("Query execution failed: %s", exc)
            raise DatabaseError(f"Query execution failed: {exc}") from exc

    def explain_query(self, sql: str) -> None:
        """Compile the query via EXPLAIN PLAN, raising on any error."""
        self.connect()
        logger.debug("EXPLAIN PLAN FOR %s", sql)
        self._execute(f"EXPLAIN PLAN FOR {sql}")

    def fetch_schema(self) -> dict[str, Any]:
        """Discover business tables, views and columns for the current schema.

        Only the schema the connection is logged into (e.g. ADMIN) is read, so
        the semantic layer never iterates over other user schemas.
        """
        self.connect()
        owner = self._current_owner()
        logger.info("Fetching schema for owner: %s", owner)
        return {owner: {"tables": self._fetch_objects(owner)}}

    def _current_owner(self) -> str:
        rows = self._query_dicts(
            "SELECT SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') AS SCHEMA_NAME FROM DUAL"
        )
        if not rows or not rows[0].get("SCHEMA_NAME"):
            raise DatabaseError("Unable to determine the current database schema.")
        return str(rows[0]["SCHEMA_NAME"]).upper()

    def close(self) -> None:
        """Close the underlying connection, if open."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _fetch_objects(self, owner: str) -> dict[str, Any]:
        rows = self._query_dicts(COLUMN_QUERY, {"owner": owner})
        tables: dict[str, Any] = {}
        for row in rows:
            name = str(row["TABLE_NAME"] or "").upper()
            object_type = str(row["TABLE_TYPE"] or "TABLE").upper()
            table_comment = row["TABLE_COMMENTS"]
            column_name = str(row["COLUMN_NAME"] or "").upper()
            column_type = str(row["DATA_TYPE"] or "").upper()
            column_comment = row["COLUMN_COMMENTS"]
            tables.setdefault(
                name,
                {"type": object_type, "description": table_comment, "columns": {}},
            )
            tables[name]["columns"][column_name] = {
                "type": column_type,
                "description": column_comment,
            }
        return tables

    def _query_dicts(
        self,
        sql: str,
        parameters: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._connection is None:
            raise DatabaseError("No active connection.")
        with self._connection.cursor() as cursor:
            cursor.execute(sql, parameters) if parameters else cursor.execute(sql)
            if cursor.description is None:
                return []
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def _execute(self, sql: str) -> None:
        if self._connection is None:
            raise DatabaseError("No active connection.")
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(sql)
        except oracledb.Error as exc:
            raise DatabaseError(f"SQL execution failed: {exc}") from exc


COLUMN_QUERY = """
    SELECT c.OWNER,
           c.TABLE_NAME,
           CASE WHEN v.VIEW_NAME IS NULL THEN 'TABLE' ELSE 'VIEW' END AS TABLE_TYPE,
           tc.COMMENTS AS TABLE_COMMENTS,
           c.COLUMN_NAME,
           c.DATA_TYPE,
           cc.COMMENTS AS COLUMN_COMMENTS
    FROM ALL_TAB_COLUMNS c
    LEFT JOIN ALL_VIEWS v
      ON c.OWNER = v.OWNER AND c.TABLE_NAME = v.VIEW_NAME
    LEFT JOIN ALL_TAB_COMMENTS tc
      ON c.OWNER = tc.OWNER AND c.TABLE_NAME = tc.TABLE_NAME
    LEFT JOIN ALL_COL_COMMENTS cc
      ON c.OWNER = cc.OWNER AND c.TABLE_NAME = cc.TABLE_NAME AND c.COLUMN_NAME = cc.COLUMN_NAME
    WHERE c.OWNER = :owner
    ORDER BY c.TABLE_NAME, c.COLUMN_ID
"""
