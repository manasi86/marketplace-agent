"""Graph node implementations for the coordinator agent."""

from collections.abc import Callable
import logging
from typing import Any

from agents.common.context import AgentContext
from agents.common.llm import LLMError, invoke_llm
from agents.common.prompts import parse_json_object
from agents.coordinator.prompts import classification_prompt
from agents.coordinator.state import FALLBACK_CATEGORY, CoordinatorState
from agents.registry import get_agent_for_raw

logger = logging.getLogger(__name__)

NodeFn = Callable[[CoordinatorState], dict[str, Any]]


def build_nodes(context: AgentContext) -> dict[str, NodeFn]:
    """Create the coordinator graph nodes closed over the shared context."""

    def classify_intent(state: CoordinatorState) -> dict[str, Any]:
        logger.info("Classifying intent for query: %s", state["user_query"])
        raw_category: str | None = None
        try:
            raw = invoke_llm(context.llm, classification_prompt(state["user_query"]))
            parsed = parse_json_object(raw)
            raw_category = parsed.get("category")
        except LLMError:
            logger.warning(
                "LLM classification failed; falling back to %s.",
                FALLBACK_CATEGORY.value,
                exc_info=True,
            )
        category_spec = str(raw_category) if raw_category else FALLBACK_CATEGORY.value
        category = get_agent_for_raw(category_spec)
        logger.info("Query classified as: %s", category.value)
        return {
            "category": category,
            "logs": [
                (
                    "classify_intent",
                    f"Classified as '{category.value}'",
                ),
            ],
        }

    return {"classify_intent": classify_intent}
