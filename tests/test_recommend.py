"""Tests for the Recommend agent's interactive planning flow."""

from typing import Any

import pytest

from agents.common.db import QueryResult
from agents.recommend.agent import (
    GOAL_PROMPT,
    _format_query_result,
    approval_detection,
    plan_to_markdown,
    run_agent,
)
from agents.recommend.prompts import (
    data_needs_prompt,
    evaluate_plan_json,
    goal_detection_prompt,
    strategist_planner_prompt,
)
from agents.recommend.state import initial_recommend_state
from tests.doubles import FakeLLM, FakeOracle, make_context

GOOD_SQL = "SELECT region, SUM(total) FROM SALES.VW_SALES_SUMMARY GROUP BY region"
GOAL_JSON = '{"goal_clear": true, "goal": "Increase Q3 revenue across Tier 2 cities"}'
GOAL_UNCLEAR = '{"goal_clear": false, "goal": null}'
GOAL_SUPPLIED = '{"goal_clear": true, "goal": "Boost repeat purchases in Tier 1"}'
DATA_NEEDS_JSON = '{"data_needs": ["total revenue by region for Q3"]}'
DATA_NEEDS_EMPTY = '{"data_needs": []}'
DATA_NEEDS_TWO = '{"data_needs": ["total revenue by region", "inventory by city"]}'

SAMPLE_SCHEMA: dict[str, Any] = {
    "SALES": {
        "tables": {
            "VW_SALES_SUMMARY": {
                "type": "VIEW",
                "description": "Sales summary",
                "columns": {
                    "REGION": {"type": "VARCHAR2", "description": "Region"},
                    "TOTAL": {"type": "NUMBER", "description": "Total"},
                },
            }
        }
    }
}


def _plan_json(
    *,
    observe_duration: str = "30 days",
    recommendations: list[str] | None = None,
) -> str:
    payload = {
        "understanding": "Restate the ask",
        "data_used": "revenue by region, Q3, tier 2",
        "key_findings": ["Assumption: margin recovery in tier 2 is viable"],
        "recommendations": recommendations or ["Expand distribution in tier 2 cities"],
        "data_gaps": "Data on tier 2 margin is sparse",
        "observe_duration": observe_duration,
    }
    import json

    return json.dumps(payload)


def _context(*, responses: list[str], explain_failures: int = 0) -> Any:
    return make_context(
        llm=FakeLLM(responses),
        connection=FakeOracle(
            fetch_schema=SAMPLE_SCHEMA,
            explain_failures=explain_failures,
            result=QueryResult(columns=["REGION"], rows=[["West", 100]]),
        ),
    )


def _inputs(replies: list[str]) -> Any:
    index = 0

    def _next(prompt: str) -> str:
        nonlocal index
        del prompt
        if index >= len(replies):
            raise AssertionError("No more scripted user inputs.")
        reply = replies[index]
        index += 1
        return reply

    return _next


# ---------------------------------------------------------------------------
# Prompt unit tests
# ---------------------------------------------------------------------------


def test_approval_detection() -> None:
    for text in ("approve", "Approved", "yes", "go ahead please", "ok"):
        assert approval_detection(text)
    for text in ("change it", "no", "maybe", ""):
        assert not approval_detection(text)


def test_goal_detection_prompt_mentions_request() -> None:
    assert "Increase sales" in goal_detection_prompt("Increase sales")


def test_data_needs_prompt_mentions_request() -> None:
    prompt = data_needs_prompt("What is total revenue?", "Grow Q3")
    assert "What is total revenue?" in prompt
    assert "Grow Q3" in prompt
    assert "Inventory — All Locations" in prompt
    assert "Sales — All" in prompt


def test_data_needs_prompt_without_goal() -> None:
    prompt = data_needs_prompt("What is total revenue?")
    assert "What is total revenue?" in prompt
    assert "Goal:" not in prompt


def test_strategist_planner_prompt_includes_data_summary() -> None:
    prompt = strategist_planner_prompt(
        "expand",
        "Grow Q3",
        "### Data need: revenue\n| REGION |\n| West |",
        change_request="Focus on pricing",
    )
    assert "Grow Q3" in prompt
    assert "Focus on pricing" in prompt
    assert "Indian Market Strategist" in prompt
    assert "Inventory — All Locations" in prompt
    assert "personal care/FMCG" in prompt
    assert "data_used" in prompt
    assert "data_gaps" in prompt
    assert "observe_duration" in prompt
    assert "### Data need: revenue" in prompt
    assert "evaluation_questions" not in prompt


def test_plan_to_markdown_renders_sections() -> None:
    parsed, ok = evaluate_plan_json(_plan_json(recommendations=["R1"]))
    assert ok
    markdown = plan_to_markdown(parsed)
    assert "## Recommended plan" in markdown
    assert "1. R1" in markdown
    assert "Observe duration" not in markdown
    assert "## Data used" in markdown
    assert "## Data gaps" in markdown
    assert "## Key findings" in markdown


def test_plan_to_markdown_with_missing_sections() -> None:
    assert plan_to_markdown({}) == "No structured plan could be rendered."


def test_plan_to_markdown_handles_string_findings() -> None:
    parsed = {
        "understanding": "u",
        "data_used": "s",
        "key_findings": "- one\n- two",
        "recommendations": ["r1"],
        "data_gaps": "r",
    }
    markdown = plan_to_markdown(parsed)
    assert "- one" in markdown
    assert "- two" in markdown


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------


def test_initial_recommend_state() -> None:
    state = initial_recommend_state("q")
    assert state["status"] == "awaiting_goal"
    assert state["needs_goal"] is False
    assert state["data_summary"] is None
    assert state["queries_executed"] == []


# ---------------------------------------------------------------------------
# Format helper tests
# ---------------------------------------------------------------------------


def test_format_query_result_basic() -> None:
    text = _format_query_result(
        "revenue by region",
        "SELECT region, total FROM sales",
        ["REGION", "TOTAL"],
        [["West", 100], ["East", 200]],
    )
    assert "revenue by region" in text
    assert "SELECT region, total FROM sales" in text
    assert "REGION | TOTAL" in text
    assert "West | 100" in text
    assert "East | 200" in text


def test_format_query_result_empty_rows() -> None:
    text = _format_query_result("empty query", "SELECT 1 FROM DUAL", ["X"], [])
    assert "No rows returned" in text


def test_format_query_result_many_rows() -> None:
    rows = [[f"row{i}"] for i in range(60)]
    text = _format_query_result("big", "SELECT x FROM t", ["X"], rows)
    assert "row0" in text
    assert "row49" in text
    assert "10 more rows" in text


def test_format_query_result_no_columns() -> None:
    text = _format_query_result("no cols", "SELECT 1", [], [[1]])
    assert "1" in text


# ---------------------------------------------------------------------------
# Integration tests — basic flow
# ---------------------------------------------------------------------------


def test_approval_immediate_no_goal_prompt() -> None:
    context = _context(responses=[GOAL_JSON, DATA_NEEDS_JSON, GOOD_SQL, _plan_json()])
    state = run_agent("Expand in tier 2", context, input_fn=_inputs(["approve"]))
    assert state["approved"] is True
    assert state["status"] == "handed_to_act"
    assert state["needs_goal"] is False
    assert state["goal"] == "Increase Q3 revenue across Tier 2 cities"
    assert state["observe_duration"] == "30 days"
    assert state["done"] is True
    assert state["data_summary"] is not None
    assert "West" in (state["data_summary"] or "")
    assert len(state["queries_executed"]) == 1
    assert any(log[0] == "recommend" for log in state["logs"])
    assert any(log[0] == "data_query" for log in state["logs"])


def test_goal_unclear_prompts_and_uses_supplied() -> None:
    context = _context(
        responses=[
            GOAL_UNCLEAR,
            GOAL_SUPPLIED,
            DATA_NEEDS_JSON,
            GOOD_SQL,
            _plan_json(),
        ]
    )
    replies = ["My goal is to boost repeat purchases in Tier 1", "approve"]
    state = run_agent("Tell me what to do", context, input_fn=_inputs(replies))
    assert state["needs_goal"] is True
    assert state["goal"] == "Boost repeat purchases in Tier 1"
    assert state["data_summary"] is not None


def test_change_request_regenerates_plan_before_approval() -> None:
    context = _context(
        responses=[
            GOAL_JSON,
            DATA_NEEDS_JSON,
            GOOD_SQL,
            _plan_json(recommendations=["First recommendation"]),
            _plan_json(recommendations=["Revised recommendation"]),
        ]
    )
    replies = ["Please focus on pricing instead", "approve"]
    state = run_agent("Recommend a plan", context, input_fn=_inputs(replies))
    assert state["approved"] is True
    assert "Revised recommendation" in (state["plan"] or "")


def test_empty_approval_reply_loops_until_approve() -> None:
    context = _context(responses=[GOAL_JSON, DATA_NEEDS_JSON, GOOD_SQL, _plan_json()])
    state = run_agent("Plan", context, input_fn=_inputs(["", "", "approve"]))
    assert state["approved"] is True


# ---------------------------------------------------------------------------
# Integration tests — data needs
# ---------------------------------------------------------------------------


def test_data_needs_determined_and_executed() -> None:
    context = _context(responses=[GOAL_JSON, DATA_NEEDS_JSON, GOOD_SQL, _plan_json()])
    state = run_agent("Plan", context, input_fn=_inputs(["approve"]))
    assert state["data_summary"] is not None
    assert "West" in (state["data_summary"] or "")
    assert len(state["queries_executed"]) == 1
    assert GOOD_SQL in state["queries_executed"]
    assert any("Determined 1 data need" in log[1] for log in state["logs"])
    assert any("Executed:" in log[1] for log in state["logs"])


def test_data_needs_empty_skips_execution() -> None:
    context = _context(responses=[GOAL_JSON, DATA_NEEDS_EMPTY, _plan_json()])
    state = run_agent("Plan", context, input_fn=_inputs(["approve"]))
    assert state["data_summary"] == "No data queries were identified."
    assert state["queries_executed"] == []
    assert any("Determined 0 data need" in log[1] for log in state["logs"])


def test_data_needs_two_queries_executed() -> None:
    context = _context(
        responses=[
            GOAL_JSON,
            DATA_NEEDS_TWO,
            GOOD_SQL,
            GOOD_SQL,
            _plan_json(),
        ]
    )
    state = run_agent("Plan", context, input_fn=_inputs(["approve"]))
    assert len(state["queries_executed"]) == 2
    assert state["data_summary"] is not None
    assert "Data need:" in (state["data_summary"] or "")


def test_data_needs_llm_error_still_produces_plan() -> None:
    context = _context(responses=[GOAL_JSON, _plan_json()])
    state = run_agent("Plan", context, input_fn=_inputs(["approve"]))
    assert state["approved"] is True
    assert state["data_summary"] == "No data queries were identified."


def test_sql_pipeline_exception_skips_need(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(responses=[GOAL_JSON, DATA_NEEDS_JSON, _plan_json()])

    def _boom(question: str, ctx: Any) -> Any:
        del question, ctx
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr("agents.recommend.agent.run_sql_pipeline", _boom)
    state = run_agent("Plan", context, input_fn=_inputs(["approve"]))
    assert state["queries_executed"] == []
    assert any("Failed:" in log[1] for log in state["logs"])


def test_sql_pipeline_error_result_skips_need() -> None:
    context = _context(responses=[GOAL_JSON, DATA_NEEDS_JSON, _plan_json()])
    # FakeOracle with execute_error will make the pipeline fail at execute step
    context.connection.execute_error = "ORA-00942: table or view does not exist"
    state = run_agent("Plan", context, input_fn=_inputs(["approve"]))
    assert state["queries_executed"] == []
    assert any("Error for" in log[1] for log in state["logs"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_planner_unparseable_falls_back_to_qualitative() -> None:
    context = _context(responses=[GOAL_JSON, DATA_NEEDS_JSON, GOOD_SQL, "not-json-at-all"])
    state = run_agent("Expand", context, input_fn=_inputs(["approve"]))
    assert state["approved"] is True
    assert state["observe_duration"] == "30 days"


def test_plan_generation_llm_error_records_error() -> None:
    context = _context(responses=[GOAL_JSON, DATA_NEEDS_EMPTY])
    state = run_agent("Plan", context, input_fn=_inputs(["approve"]))
    assert state["approved"] is True
    assert "Plan generation failed" in (state["plan"] or "")


def test_goal_prompt_text() -> None:
    assert "goal" in GOAL_PROMPT.lower()


def test_empty_supplied_goal_reprompts() -> None:
    responses = [GOAL_UNCLEAR, GOAL_SUPPLIED, DATA_NEEDS_JSON, GOOD_SQL, _plan_json()]
    context = _context(responses=responses)
    replies = ["", "My goal is to boost repeat purchases in Tier 1", "approve"]
    state = run_agent("Plan", context, input_fn=_inputs(replies))
    assert state["needs_goal"] is True
    assert state["goal"] == "Boost repeat purchases in Tier 1"


def test_goal_detection_llm_error_uses_supplied_goal() -> None:
    context = _context(responses=[GOAL_UNCLEAR])
    replies = ["Boost repeat purchases", "approve"]
    state = run_agent("Plan", context, input_fn=_inputs(replies))
    assert state["needs_goal"] is True
    assert state["goal"] == "Boost repeat purchases"


def test_goal_unclear_but_alternative_parsed() -> None:
    ambiguous = '{"goal_clear": false, "goal": "Grow Q3 margin"}'
    context = _context(responses=[ambiguous, DATA_NEEDS_JSON, GOOD_SQL, _plan_json()])
    state = run_agent("Plan", context, input_fn=_inputs(["approve"]))
    assert state["goal"] == "Grow Q3 margin"


def test_detect_goal_empty_clear_goal_falls_through() -> None:
    from agents.recommend.agent import _detect_goal

    llm = FakeLLM(['{"goal_clear": true, "goal": "  "}'])
    assert _detect_goal("q", llm, candidate=None) is None


def test_act_handoff_exception_records_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(responses=[GOAL_JSON, DATA_NEEDS_JSON, GOOD_SQL, _plan_json()])

    def _boom(plan: str, ctx: Any) -> Any:
        del plan, ctx
        raise RuntimeError("act exploded")

    monkeypatch.setattr("agents.act.agent.run_agent", _boom)
    state = run_agent("Plan", context, input_fn=_inputs(["approve"]))
    assert state["approved"] is True
    assert "Act handoff failed" in (state["error"] or "")


def test_act_handoff_without_error_keeps_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(responses=[GOAL_JSON, DATA_NEEDS_JSON, GOOD_SQL, _plan_json()])

    def _clean(plan: str, ctx: Any) -> Any:
        del plan, ctx
        return {"logs": [], "error": None, "done": True}

    monkeypatch.setattr("agents.act.agent.run_agent", _clean)
    state = run_agent("Plan", context, input_fn=_inputs(["approve"]))
    assert state["approved"] is True
    assert state["status"] == "handed_to_act"
    assert state["error"] is None


def test_determine_data_needs_returns_list() -> None:
    from agents.recommend.agent import _determine_data_needs

    llm = FakeLLM([DATA_NEEDS_JSON])
    state = initial_recommend_state("q")
    needs = _determine_data_needs("q", "goal", llm, state)
    assert needs == ["total revenue by region for Q3"]


def test_determine_data_needs_llm_error_returns_empty() -> None:
    from agents.recommend.agent import _determine_data_needs

    llm = FakeLLM([])
    state = initial_recommend_state("q")
    needs = _determine_data_needs("q", "goal", llm, state)
    assert needs == []


def test_execute_data_needs_no_needs() -> None:
    from agents.recommend.agent import _execute_data_needs

    context = _context(responses=[])
    state = initial_recommend_state("q")
    summary = _execute_data_needs([], context, state)
    assert summary == "No data queries were identified."
