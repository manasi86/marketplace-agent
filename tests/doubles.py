"""Shared fakes for SQL generator tests."""

from typing import Any, cast

from langchain_core.language_models import LanguageModelLike

from lib.sql_generator.config import Settings
from lib.sql_generator.context import AgentContext
from lib.sql_generator.db import DatabaseError, OracleConnection, QueryResult
from lib.sql_generator.llm import LLMError
from lib.sql_generator.semantic import SemanticContext


class FakeLLM:
    """Stub LLM that returns pre-scripted string responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise LLMError("No scripted responses left.")
        return self._responses.pop(0)


class FakeOracle:
    """Stub Oracle connection for driving the agent pipeline."""

    def __init__(
        self,
        *,
        connected: bool = True,
        fetch_schema: dict[str, Any] | None = None,
        explain_failures: int = 0,
        execute_error: str | None = None,
        result: QueryResult | None = None,
    ) -> None:
        self.connected = connected
        self._fetch_schema = fetch_schema or {}
        self.explain_failures = explain_failures
        self.execute_error = execute_error
        self.result = result
        self.explain_calls = 0
        self.executed_sql: list[str] = []

    def check_connection(self) -> bool:
        return self.connected

    def fetch_schema(self) -> dict[str, Any]:
        return self._fetch_schema

    def explain_query(self, sql: str) -> None:
        self.explain_calls += 1
        if self.explain_calls <= self.explain_failures:
            raise DatabaseError("ORA-00904: invalid identifier from fake")

    def execute_query(self, sql: str) -> QueryResult:
        self.executed_sql.append(sql)
        if self.execute_error is not None:
            raise DatabaseError(self.execute_error)
        if self.result is not None:
            return self.result
        return QueryResult(columns=["REGION"], rows=[["West", 100]])


def make_context(
    *,
    settings: Settings | None = None,
    llm: FakeLLM | None = None,
    connection: FakeOracle | None = None,
    semantic: SemanticContext | None = None,
) -> AgentContext:
    """Build an AgentContext wired to fakes, defaulting sensibly."""
    resolved_settings = settings or Settings(sql_gen_api_key="test-key", langfuse_enabled=False)
    return AgentContext(
        settings=resolved_settings,
        llm=cast(LanguageModelLike, llm or FakeLLM()),
        connection=cast(OracleConnection, connection or FakeOracle()),
        semantic=semantic or SemanticContext(),
    )
