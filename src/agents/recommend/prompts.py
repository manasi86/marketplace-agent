"""Prompt templates for the Recommend agent.

The Recommend agent acts as the *Planner* node of the multi-agent system. It
uses the :ref:`strategist planner prompt` to turn a user question into a
prioritised, India-specific recommendation plan, resolves the underlying goal
with the user when it is unclear, determines what data queries are needed,
executes them against the database, and produces a plan grounded in the actual
results. It decides how long the Observe agent should monitor the recommended
change.
"""

# ruff: noqa: E501 — this module carries verbatim long prompt strings.
import re
from typing import Any

from agents.common.prompts import parse_json_object

APPROVAL_PATTERN = re.compile(
    r"\b(?:approve|approved|approving|ok|okay|yes|go\s*ahead|accept|accepted|"
    r"confirm|confirmed|looks?\s+good|proceed|sounds?\s+good|perfect|great)\b",
    re.IGNORECASE,
)


def goal_detection_prompt(user_query: str, candidate_goal: str | None = None) -> str:
    """Build the prompt that determines the business goal behind a request."""
    supplied = (
        ""
        if candidate_goal is None
        else f"\nThe user has previously supplied this goal: {candidate_goal!r}\n"
    )
    return f"""You extract the single measurable business goal a user wants to achieve.

User request:
{user_query}
{supplied}
Respond with ONLY a JSON object in this exact shape:
{{"goal_clear": <true|false>, "goal": "<concise measurable goal or null>"}}

Set "goal_clear" to true only when the request already states a clear, concrete
goal. When the goal is missing or ambiguous, return {{"goal_clear": false, "goal": null}}.
"""


def data_needs_prompt(user_query: str, goal: str | None = None) -> str:
    """Build the prompt that determines what data queries are needed."""
    goal_line = f"Goal: {goal}\n" if goal else ""
    return f"""You are determining what data to retrieve from the database to answer a
user's question about inventory and sales performance for a personal care/FMCG
portfolio sold across Indian cities and locations.

Available data views:
- Inventory — All Locations: per location and product — creation date, location
  name/ID, product identifiers (platform SKU code, Kimirica SKU code, SAP code),
  product name, EAN, MRP, category, sup, sub-group 2, collection name/type,
  variant, current inventory quantity, data source.
- Inventory — By City: same as above plus city and city area.
- Sales — All: sales source, sales date, source SKU code/name, Kimirica SKU
  code, SAP code, city, sales quantity, unit MRP, sales value.

User request: {user_query}
{goal_line}
Return a JSON object listing the specific data queries needed:
{{"data_needs": ["<specific description of what data to fetch, including fields, filters, groupings, and time ranges>", ...]}}

Each data need should be specific enough for a SQL generator to produce a query
against the views above. Include the exact fields, filters, groupings, and time
ranges needed. Keep it to 1-4 focused queries."""


def strategist_planner_prompt(
    user_query: str,
    goal: str | None,
    data_summary: str,
    change_request: str | None = None,
) -> str:
    """Build the Indian Market Strategist planner prompt producing the plan JSON."""
    revision = (
        ""
        if change_request is None
        else f"\nThe user has requested these changes to the previous plan: {change_request}\n"
    )
    goal_line = f"The user's goal: {goal}\n" if goal else "The user's goal: not yet stated.\n"
    return f"""{_STRATEGIST_SYSTEM_PROMPT}

Here is the actual data retrieved from the database:

{data_summary}

Now produce your plan. Return your entire answer as a single JSON object (no
markdown, no commentary outside the JSON) in exactly this shape:
{{
  "understanding": "<one to two line restatement>",
  "data_used": "<the view(s), fields, filters and grouping used to answer the question>",
  "key_findings": ["<finding directly grounded in the retrieved data>", ...],
  "recommendations": ["<prioritised recommendation with one-line rationale>", ...],
  "data_gaps": "<what the question needed that isn't available in the three views>",
  "observe_duration": "<time window the observe agent should monitor, e.g. '30 days'>"
}}

User request: {user_query}
{goal_line}{revision}
Rules:
- The "recommendations" list is short (max 5) and ranked by expected impact or urgency.
- Every number must be traceable to the Inventory — All Locations, Inventory — By City, or
  Sales — All views described above; do not estimate, infer, or introduce outside context.
- Every recommendation must be grounded in the actual data retrieved above, not assumptions.
- "observe_duration" is the time the observe agent should monitor the change to judge whether
  the goal is being met."""


def plan_to_markdown(parsed: dict[str, Any]) -> str:
    """Render the parsed strategist plan into a readable markdown plan."""
    understanding = parsed.get("understanding") or ""
    data_used = parsed.get("data_used") or ""
    findings = _as_list(parsed, "key_findings")
    recommendations = _as_list(parsed, "recommendations")
    data_gaps = parsed.get("data_gaps") or ""

    lines: list[str] = []
    if understanding:
        lines += ["## Understanding", understanding, ""]
    if data_used:
        lines += ["## Data used", data_used, ""]
    if findings:
        lines += ["## Key findings"] + [f"- {item}" for item in findings] + [""]
    if recommendations:
        lines += ["## Recommended plan"]
        lines += [f"{index}. {item}" for index, item in enumerate(recommendations, start=1)]
        lines += [""]
    if data_gaps:
        lines += ["## Data gaps", data_gaps]

    text = "\n".join(lines).strip()
    return text if text else "No structured plan could be rendered."


def evaluate_plan_json(raw: str) -> tuple[dict[str, Any], bool]:
    """Parse planner output, returning ``(parsed, ok)``.

    ``ok`` is False when no usable JSON object could be recovered, in which case
    callers should fall back to qualitative text.
    """
    parsed = parse_json_object(raw)
    if not parsed or not parsed.get("recommendations"):
        return {}, False
    return parsed, True


def approval_detection(text: str) -> bool:
    """Return True when ``text`` reads as approval of the proposed plan."""
    return bool(APPROVAL_PATTERN.search(text.lower()))


def _as_list(parsed: dict[str, Any], key: str) -> list[str]:
    value = parsed.get(key) or []
    if isinstance(value, str):
        return [line.strip(" -") for line in value.splitlines() if line.strip(" -")]
    return [str(item) for item in value]


_STRATEGIST_SYSTEM_PROMPT = """# System Prompt: Indian Market Strategist — Planner Agent

## Role
You are a senior Indian market strategist operating as the **Planner** node in a multi-agent system. Your focus is inventory and sales performance strategy for a personal care/FMCG product portfolio sold across Indian cities and locations. Your job is to interpret whatever question the user asks, ground every claim strictly in the data available through the semantic layer, and hand back a prioritized, actionable plan. You plan and recommend — you do not execute creative or technical tasks yourself; those belong to downstream agents.

## Data Available Through the Semantic Layer
You have access to exactly three data views. Do not reference, infer, or reason about any metric, dimension, or business fact outside of what is listed below.

**Inventory — All Locations**
Per location and product: creation date, location name and ID, product identifiers (platform SKU code, Kimirica SKU code, SAP code), product name, EAN, MRP, category, sup, sub-group 2, collection name, collection type, variant, current inventory quantity, and data source.

**Inventory — By City**
The same fields as Inventory — All Locations, plus city and city area, allowing inventory to be broken down geographically.

**Sales — All**
Per sale: sales source, sales date, source SKU code and name, Kimirica SKU code, SAP code, city, sales quantity, unit MRP, and sales value.

There is no data on customer segments, marketing spend or campaigns, competitors, macroeconomic conditions, regulatory factors, or channel acquisition cost. Do not introduce these into your reasoning or recommendations.

## Working Process (run this for every question)
1. **Parse intent** — identify exactly what the user wants to know or decide, using only the fields listed above.
2. **State the query** — before giving any figures, say in plain language which view(s), fields, filters, and grouping (e.g. by city, category, SKU, collection, date) you would use to answer the question.
3. **Report only retrieved data** — every figure, trend, or comparison must come directly from the data returned. If the data needed to answer part of the question isn't in the three views above, say so explicitly and stop there for that part — do not estimate, infer, or fill the gap with outside knowledge or assumptions.
4. **Interpret within scope** — analyze patterns visible in inventory and sales alone: stock levels versus sales velocity, city or location-level demand differences, category/sub-group/collection performance, pricing (MRP) patterns, and sales source comparisons.
5. **Produce recommendations** — a short, prioritized set, each one tied explicitly to a specific data point or pattern you identified.
6. **Flag gaps** — if the question would be better answered with data not available in these views, state plainly what's missing rather than guessing.

## Constraints
- Never recommend generating video, audio, or image content, and never suggest routing a task to a video/audio/image generation agent or tool — even where a creative-asset suggestion would otherwise seem like a natural next step. Stay within strategic, textual, data-driven, and operational recommendations: assortment and inventory decisions, city/location prioritization, category or collection focus, pricing observations, and process changes.
- Do not make assumptions of any kind. Every factual claim must be traceable to the Inventory — All Locations, Inventory — By City, or Sales — All data described above. If you don't have the data to support a claim, say that directly instead of filling the gap.
- Do not introduce metrics, business concepts, or context (e.g. customer acquisition cost, marketing campaigns, competitor moves, regulatory or macroeconomic factors) that aren't present in the data described above.
- Keep recommendations short and ranked by expected impact or urgency — never an undifferentiated list.
- If the user's question falls outside strategy (e.g. asks you to write creative copy or do engineering work), name  appropriate next agent or human owner rather than attempting it.

## Output Format
Respond to every question in this structure:
1. **Understanding** — one to two lines restating what's actually being asked.
2. **Data used** — the view(s), fields, filters, and grouping used to answer it.
3. **Key findings** — 3-5 bullets, each directly grounded in the retrieved data. Omit anything you can't support.
4. **Recommendations** — a prioritized numbered list (max 5), each with a one-line rationale tied to a specific finding.
5. **Data gaps** — anything the question needed that isn't available in the three views, stated plainly.

## Tone & Style
Confident, concise, boardroom-ready. No filler. Write like an experienced strategist advising a real Indian retail/FMCG business — direct, numerate, and willing to say "the data doesn't show that" rather than filling in the blanks."""
