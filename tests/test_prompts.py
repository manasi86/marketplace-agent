"""Tests for the prompt templates."""

from agents.sql_generator.prompts import sql_generation_prompt, sql_generation_system_message


def test_sql_generation_prompt_without_error() -> None:
    prompt = sql_generation_prompt(
        user_query="Show total sales by region",
        semantic_context="Schema: SALES\n  VIEW VW_SALES_SUMMARY",
    )
    assert "Show total sales by region" in prompt
    assert "Schema: SALES" in prompt
    assert "Previous attempt FAILED" not in prompt
    assert "Do not include any explanatory text" in prompt


def test_sql_generation_prompt_includes_validation_error() -> None:
    prompt = sql_generation_prompt(
        user_query="Show total sales by region",
        semantic_context="Schema: SALES",
        validation_error="ORA-00904: invalid identifier",
    )
    assert "ORA-00904" in prompt
    assert "Your previous attempt FAILED validation" in prompt


def test_sql_generation_system_message() -> None:
    message = sql_generation_system_message()
    assert "SELECT" in message
    assert message != ""
