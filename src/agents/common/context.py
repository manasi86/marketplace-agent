"""Runtime context construction shared by every agent."""

from dataclasses import dataclass

from langchain_core.language_models import LanguageModelLike

from agents.common.config import Settings, get_settings
from agents.common.db import OracleConnection
from agents.common.llm import get_llm
from agents.common.semantic import SemanticContext


@dataclass
class AgentContext:
    """Concrete dependencies bound at startup and used by all agents."""

    settings: Settings
    llm: LanguageModelLike
    connection: OracleConnection
    semantic: SemanticContext


def build_context(settings: Settings | None = None) -> AgentContext:
    """Construct an AgentContext, resolving default settings if not supplied."""
    resolved = settings or get_settings()
    return AgentContext(
        settings=resolved,
        llm=get_llm(resolved),
        connection=OracleConnection(resolved),
        semantic=SemanticContext(),
    )
