"""Tests for runtime context assembly and package exports."""

import pytest

from lib.sql_generator import build_context, build_graph, initial_state, run_agent
from lib.sql_generator.config import Settings, get_settings
from lib.sql_generator.context import AgentContext
from lib.sql_generator.context import build_context as _build_context
from lib.sql_generator.db import OracleConnection
from lib.sql_generator.semantic import SemanticContext


def test_build_context_with_explicit_settings() -> None:
    settings = Settings(sql_gen_api_key="test-key", langfuse_enabled=False)
    context = _build_context(settings)
    assert isinstance(context, AgentContext)
    assert context.settings is settings
    assert isinstance(context.connection, OracleConnection)
    assert isinstance(context.semantic, SemanticContext)
    assert context.llm is not None


def test_build_context_resolves_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQL_GEN_API_KEY", "env-key")
    get_settings.cache_clear()
    try:
        context = _build_context()
        assert context.settings.sql_gen_api_key == "env-key"
        assert context.llm is not None
    finally:
        get_settings.cache_clear()


def test_package_exports() -> None:
    assert callable(build_context)
    assert callable(build_graph)
    assert callable(run_agent)
    assert callable(initial_state)
