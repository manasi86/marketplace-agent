"""Recommend agent (Planner): interactive India-specific strategy planning.

The agent runs a short conversation with the user: it resolves the underlying
goal (prompting when unclear), determines what data queries are needed to answer
the question, executes them against the database, and produces a prioritised
plan grounded in the actual results using the Indian Market Strategist planner
prompt. It decides how long the Observe agent should monitor the recommended
change. It only hands the approved plan to the Act agent once the user says
something like "approve"; any other reply is treated as a request to revise the
plan.
"""

from collections.abc import Callable
import logging
from typing import Any, cast

from agents.common.context import AgentContext
from agents.common.llm import LLMError, invoke_llm
from agents.common.observability import observe_run
from agents.common.prompts import parse_json_object
from agents.recommend.prompts import (
    approval_detection,
    data_needs_prompt,
    evaluate_plan_json,
    goal_detection_prompt,
    plan_to_markdown,
    strategist_planner_prompt,
)
from agents.recommend.state import RecommendState, initial_recommend_state
from agents.sql_generator.graph import run_agent as run_sql_pipeline

logger = logging.getLogger(__name__)

APPROVAL_PROMPT = (
    "\nReply 'approve' to proceed to the Act agent, or describe the changes "
    "you want made to the plan: "
)
GOAL_PROMPT = "Please describe your goal for this recommendation: "
DEFAULT_OBSERVE_DURATION = "30 days"
_MAX_ROWS_PER_QUERY = 50


@observe_run("recommend_agent")
def run_agent(
    user_query: str,
    context: AgentContext,
    *,
    input_fn: Callable[[str], str] | None = None,
) -> RecommendState:
    """Run the interactive recommend conversation and return the final state."""
    interact = input_fn or context.input_fn
    state = initial_recommend_state(user_query)

    goal = _resolve_goal(user_query, context.llm, interact, state)

    data_needs = _determine_data_needs(user_query, goal, context.llm, state)
    data_summary = _execute_data_needs(data_needs, context, state)

    change_request: str | None = None
    while True:
        plan, parsed, observe_duration = _plan(
            user_query, goal, data_summary, context.llm, change_request=change_request
        )
        _apply_plan(state, plan, parsed, observe_duration, data_summary)
        print(plan)
        print("\nObserve duration:", state["observe_duration"])

        reply = interact(APPROVAL_PROMPT).strip()
        if not reply:
            continue

        if approval_detection(reply):
            state["approved"] = True
            state["status"] = "approved"
            state["error"] = None
            break

        logger.info("Approval not given; revising plan based on: %s", reply)
        change_request = reply

    state["goal"] = goal
    state["plan"] = _final_plan(state)
    state["done"] = True
    _hand_off_to_act(state, context)
    return state


def _resolve_goal(
    user_query: str,
    llm: Any,
    input_fn: Callable[[str], str],
    state: RecommendState,
) -> str:
    """Return the user's goal, prompting them until it is clear."""
    goal = _detect_goal(user_query, llm, candidate=None)
    while goal is None:
        supplied = input_fn(GOAL_PROMPT).strip()
        if not supplied:
            continue
        goal = _detect_goal(user_query, llm, candidate=supplied)
        state["needs_goal"] = True
    state["goal"] = goal
    state["logs"].append(("recommend", f"Resolved goal: {goal}"))
    return goal


def _detect_goal(
    user_query: str,
    llm: Any,
    *,
    candidate: str | None,
) -> str | None:
    """Ask the LLM whether the goal is clear; return it, or None to prompt."""
    prompt = goal_detection_prompt(user_query, candidate)
    try:
        raw = invoke_llm(llm, prompt)
    except LLMError:
        logger.warning("Goal detection failed; using supplied goal if any.", exc_info=True)
        return candidate or None
    parsed = parse_json_object(raw)
    if parsed.get("goal_clear") is True:
        goal = str(parsed.get("goal") or "").strip()
        if goal:
            return goal
    alternative = str(parsed.get("goal") or "").strip()
    if alternative:
        return alternative
    return candidate or None


def _determine_data_needs(
    user_query: str,
    goal: str,
    llm: Any,
    state: RecommendState,
) -> list[str]:
    """Ask the LLM what data queries are needed to answer the question."""
    prompt = data_needs_prompt(user_query, goal)
    try:
        raw = invoke_llm(llm, prompt)
    except LLMError as exc:
        logger.error("Data needs determination failed: %s", exc)
        state["logs"].append(("recommend", f"Data needs determination failed: {exc}"))
        return []
    parsed = parse_json_object(raw)
    needs = parsed.get("data_needs") or []
    result = [str(n) for n in needs if str(n).strip()]
    state["logs"].append(("recommend", f"Determined {len(result)} data need(s)"))
    return result


def _execute_data_needs(
    data_needs: list[str],
    context: AgentContext,
    state: RecommendState,
) -> str:
    """Execute each data need via the SQL pipeline and return a formatted summary."""
    if not data_needs:
        return "No data queries were identified."
    summaries: list[str] = []
    for need in data_needs:
        try:
            result = run_sql_pipeline(need, context)
        except Exception as exc:
            logger.warning("SQL pipeline failed for '%s': %s", need, exc)
            state["logs"].append(("data_query", f"Failed: {need} — {exc}"))
            continue
        if result.get("error") is not None:
            state["logs"].append(("data_query", f"Error for '{need}': {result['error']}"))
            continue
        columns = result.get("query_columns") or []
        rows = result.get("query_rows") or []
        sql = str(result.get("sql_query") or "")
        summary = _format_query_result(need, sql, columns, rows)
        summaries.append(summary)
        state["queries_executed"].append(sql)
        state["logs"].append(("data_query", f"Executed: {need} ({len(rows)} rows)"))
    return "\n\n".join(summaries) if summaries else "No data could be retrieved."


def _format_query_result(need: str, sql: str, columns: list[str], rows: list[list[Any]]) -> str:
    """Format a single query result into a readable summary."""
    lines: list[str] = [f"### Data need: {need}", f"```sql\n{sql}\n```"]
    if not rows:
        lines.append("No rows returned.")
    else:
        header = " | ".join(columns)
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows[:_MAX_ROWS_PER_QUERY]:
            lines.append(" | ".join(str(v) for v in row))
        if len(rows) > _MAX_ROWS_PER_QUERY:
            lines.append(f"... ({len(rows) - _MAX_ROWS_PER_QUERY} more rows)")
    return "\n".join(lines)


def _plan(
    user_query: str,
    goal: str,
    data_summary: str,
    llm: Any,
    *,
    change_request: str | None,
) -> tuple[str, dict[str, Any], str | None]:
    """Call the strategist planner, returning the rendered plan and its fields."""
    parsed: dict[str, Any] = {}
    prompt = strategist_planner_prompt(user_query, goal, data_summary, change_request)
    try:
        raw = invoke_llm(llm, prompt)
    except LLMError as exc:
        logger.error("Plan generation failed: %s", exc)
        return f"Plan generation failed: {exc}", parsed, None
    parsed, ok = evaluate_plan_json(raw)
    if not ok:
        logger.warning("Planner JSON unparseable; falling back to qualitative text.")
        return raw.strip() or f"No structured plan was produced for: {user_query}", {}, None
    observe_duration = str(parsed.get("observe_duration") or DEFAULT_OBSERVE_DURATION)
    return plan_to_markdown(parsed), parsed, observe_duration


def _apply_plan(
    state: RecommendState,
    plan: str,
    parsed: dict[str, Any],
    observe_duration: str | None,
    data_summary: str,
) -> None:
    """Merge a freshly generated plan into the conversation state."""
    state["plan"] = plan
    state["_last_parsed"] = parsed
    state["observe_duration"] = observe_duration or DEFAULT_OBSERVE_DURATION
    state["data_summary"] = data_summary
    state["status"] = "awaiting_approval"


def _final_plan(state: RecommendState) -> str:
    parsed = state.get("_last_parsed")
    if parsed:
        return plan_to_markdown(parsed)
    return state.get("plan") or ""


def _hand_off_to_act(state: RecommendState, context: AgentContext) -> None:
    """Pass the approved plan to the Act agent and merge its outcome."""
    from agents.act.agent import run_agent as run_act

    logger.info("Approved; handing plan to Act agent.")
    try:
        result = cast("dict[str, Any]", run_act(state["plan"] or "", context))
    except Exception as exc:
        logger.error("Act handoff failed: %s", exc)
        state["error"] = f"Act handoff failed: {exc}"
        return
    for entry in result.get("logs", []) or []:
        state["logs"].append(entry)
    if result.get("error") is not None:
        state["error"] = result.get("error")
    state["status"] = "handed_to_act"


__all__ = [
    "APPROVAL_PROMPT",
    "GOAL_PROMPT",
    "approval_detection",
    "plan_to_markdown",
    "run_agent",
]
