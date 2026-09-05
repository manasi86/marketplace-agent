"""Tests for the prompt templates."""

from datetime import date

from lib.sql_generator.prompts import (
    _render_entities,
    intent_prompt,
    sql_generation_prompt,
    sql_generation_system_message,
)


def test_intent_prompt_contains_question_and_json_shape() -> None:
    prompt = intent_prompt("Show total sales by region")
    assert "Show total sales by region" in prompt
    assert '"intent"' in prompt
    assert '"entities"' in prompt
    assert '"schema_hint"' in prompt


def test_sql_generation_prompt_without_error() -> None:
    prompt = sql_generation_prompt(
        intent="aggregate sales",
        entities={"region": "West"},
        semantic_context="Schema: SALES\n  VIEW VW_SALES_SUMMARY",
    )
    assert "aggregate sales" in prompt
    assert "Schema: SALES" in prompt
    assert "Previous attempt FAILED" not in prompt
    assert "Do not include any explanatory text" in prompt


def test_sql_generation_prompt_includes_validation_error() -> None:
    prompt = sql_generation_prompt(
        intent="aggregate sales",
        entities={},
        semantic_context="Schema: SALES",
        validation_error="ORA-00904: invalid identifier",
    )
    assert "ORA-00904" in prompt
    assert "Your previous attempt FAILED validation" in prompt


def test_sql_generation_prompt_no_dates() -> None:
    prompt = sql_generation_prompt(
        intent="aggregate sales",
        entities={},
        semantic_context="Schema: SALES",
    )
    assert "Date range:" not in prompt


def test_sql_generation_prompt_with_date_range() -> None:
    prompt = sql_generation_prompt(
        intent="sales by day",
        entities={"date_range": "2025-01-01 to 2025-12-31"},
        semantic_context="Schema: SALES",
        date_start=date(2025, 1, 1),
        date_end=date(2025, 12, 31),
    )
    assert "Date range: filter by the date using" in prompt
    assert "TO_DATE('2025-01-01', 'YYYY-MM-DD')" in prompt
    assert "TO_DATE('2025-12-31', 'YYYY-MM-DD')" in prompt


def test_sql_generation_prompt_with_single_date() -> None:
    prompt = sql_generation_prompt(
        intent="sales today",
        entities={"date_range": "2025-06-01"},
        semantic_context="Schema: SALES",
        date_start=date(2025, 6, 1),
        date_end=date(2025, 6, 1),
    )
    assert "TO_DATE('2025-06-01', 'YYYY-MM-DD')" in prompt


def test_render_entities_empty() -> None:
    assert _render_entities({}) == "none"


def test_render_entities_non_empty() -> None:
    rendered = _render_entities({"region": "West", "year": 2026})
    assert "region='West'" in rendered
    assert "year=2026" in rendered


def test_sql_generation_system_message() -> None:
    message = sql_generation_system_message()
    assert "SELECT" in message
    assert message != ""
