"""Runtime context shared by every graph node."""

from dataclasses import dataclass

from langchain_core.language_models import LanguageModelLike

from lib.sql_generator.config import Settings, get_settings
from lib.sql_generator.db import OracleConnection
from lib.sql_generator.llm import get_llm
from lib.sql_generator.semantic import SemanticContext


@dataclass
class AgentContext:
    """Concrete dependencies bound at startup and used by all nodes."""

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
