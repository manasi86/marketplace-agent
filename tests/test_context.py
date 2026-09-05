"""Tests for runtime context assembly and package exports."""

import pytest

from agents.common import AgentContext, build_context
from agents.common.config import Settings, get_settings
from agents.common.db import OracleConnection
from agents.common.semantic import SemanticContext
from agents.sql_generator import build_graph, initial_state, run_agent


def test_build_context_with_explicit_settings() -> None:
    settings = Settings(sql_gen_api_key="test-key", langfuse_enabled=False)
    context = build_context(settings)
    assert isinstance(context, AgentContext)
    assert context.settings is settings
    assert isinstance(context.connection, OracleConnection)
    assert isinstance(context.semantic, SemanticContext)
    assert context.llm is not None


def test_build_context_resolves_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQL_GEN_API_KEY", "env-key")
    get_settings.cache_clear()
    try:
        context = build_context()
        assert context.settings.sql_gen_api_key == "env-key"
        assert context.llm is not None
    finally:
        get_settings.cache_clear()


def test_package_exports() -> None:
    assert callable(build_context)
    assert callable(build_graph)
    assert callable(run_agent)
    assert callable(initial_state)
