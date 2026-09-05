"""Tests for the semantic layer."""

from typing import Any, cast

import pytest

from lib.sql_generator.db import OracleConnection
from lib.sql_generator.semantic import _MAX_FORMATTED_CHARS, SemanticContext


class _FakeConnection:
    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema
        self.fetch_calls = 0

    def fetch_schema(self) -> dict[str, Any]:
        self.fetch_calls += 1
        return self._schema


def _sample_schema() -> dict[str, Any]:
    return {
        "SALES": {
            "tables": {
                "VW_SALES_SUMMARY": {
                    "type": "VIEW",
                    "description": "Total sales per region",
                    "columns": {
                        "REGION": {"type": "VARCHAR2", "description": "Region name"},
                        "TOTAL": {"type": "NUMBER", "description": None},
                    },
                },
            }
        },
        "INVENTORY": {"tables": {}},
    }


def test_metadata_raises_before_discovery() -> None:
    semantic = SemanticContext()
    with pytest.raises(RuntimeError, match="not been run"):
        semantic.metadata  # noqa: B018


def test_discover_caches_metadata() -> None:
    schema = _sample_schema()
    connection = _FakeConnection(schema)
    semantic = SemanticContext()
    first = semantic.discover(cast(OracleConnection, connection))
    second = semantic.discover(cast(OracleConnection, connection))
    assert first == schema
    assert second is first
    assert connection.fetch_calls == 1


def test_format_for_prompt_empty_metadata() -> None:
    semantic = SemanticContext()
    semantic._metadata = {}
    assert semantic.format_for_prompt() == "No tables or views found in the database."


def test_format_for_prompt_full() -> None:
    semantic = SemanticContext()
    semantic._metadata = _sample_schema()
    rendered = semantic.format_for_prompt()
    assert "Schema: INVENTORY" not in rendered
    assert "Schema: SALES" in rendered
    assert "VIEW VW_SALES_SUMMARY: Total sales per region" in rendered
    assert "- REGION (VARCHAR2): Region name" in rendered
    assert "- TOTAL (NUMBER): no comment" in rendered


def test_format_for_prompt_schema_hint_match() -> None:
    semantic = SemanticContext()
    semantic._metadata = _sample_schema()
    rendered = semantic.format_for_prompt(schema_hint="sales")
    assert "Schema: SALES" in rendered
    assert "TOTAL" in rendered


def test_format_for_prompt_schema_hint_no_match_falls_back() -> None:
    semantic = SemanticContext()
    semantic._metadata = _sample_schema()
    rendered = semantic.format_for_prompt(schema_hint="finance")
    assert "Schema: SALES" in rendered


def test_format_for_prompt_truncates_oversized_output() -> None:
    table = {
        "type": "TABLE",
        "description": None,
        "columns": {f"C{i}": {"type": "VARCHAR2", "description": "x"} for i in range(3000)},
    }
    semantic = SemanticContext()
    semantic._metadata = {"HUGE": {"tables": {"T": table}}}
    rendered = semantic.format_for_prompt()
    assert len(rendered) <= _MAX_FORMATTED_CHARS + len(
        "\n... (semantic layer truncated at 20000 chars; provide a schema hint to narrow it)"
    )
    assert "semantic layer truncated" in rendered
    assert rendered.endswith("it)")
