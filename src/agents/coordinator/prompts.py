"""Prompt template for coordinator intent classification."""

from agents.registry import AgentCategory

CATEGORY_LABELS = ", ".join(f"'{category.value}'" for category in AgentCategory)

CATEGORY_DESCRIPTIONS = {
    AgentCategory.ACT: "perform an action or mutation (create, update, delete, apply)",
    AgentCategory.RETRIEVE: "fetch or query data (select, list, show, count, get)",
    AgentCategory.RECOMMEND: "suggest the best option given criteria or preferences",
    AgentCategory.OBSERVE: "monitor, track or watch something over time",
}


def classification_prompt(user_query: str) -> str:
    """Build the prompt that classifies a user query into one intent category."""
    category_guide = "\n".join(
        f"- {category.value}: {CATEGORY_DESCRIPTIONS[category]}" for category in AgentCategory
    )
    return f"""You are a routing coordinator. Classify the user's request into exactly one of:
    {CATEGORY_LABELS}

    Category guide:
    {category_guide}

    User request:
    {user_query}

    Respond with ONLY a JSON object in this exact shape:
    {{"category": "<one of the labels above>"}}
    """
