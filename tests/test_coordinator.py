"""Tests for the coordinator agent: classification and routing."""

from typing import Any

import pytest

from agents.common.db import QueryResult
from agents.coordinator.graph import build_graph, classify_query, run_agent
from agents.coordinator.nodes import build_nodes
from agents.coordinator.prompts import classification_prompt
from agents.coordinator.state import initial_coordinator_state
from agents.registry import AgentCategory
from tests.doubles import FakeLLM, FakeOracle, make_context

INTENT_JSON = '{"intent": "aggregate sales", "entities": {}, "schema_hint": "SALES"}'
GOOD_SQL = "SELECT region, SUM(total) FROM SALES.VW_SALES_SUMMARY GROUP BY region"
ANSWER = "Total sales for the West region were 100 units."

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


def test_classification_prompt_lists_all_categories() -> None:
    prompt = classification_prompt("Show me sales")
    for category in AgentCategory:
        assert f"'{category.value}'" in prompt
    assert "Show me sales" in prompt


def test_initial_coordinator_state() -> None:
    state = initial_coordinator_state("Hello")
    assert state == {
        "user_query": "Hello",
        "logs": [],
        "error": None,
        "done": False,
        "category": None,
    }


def test_build_graph_returns_compiled_graph() -> None:
    graph = build_graph(make_context())
    assert callable(graph.invoke)
    assert "classify_intent" in graph.nodes


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"category": "act"}', AgentCategory.ACT),
        ('{"category": "RETRIEVE"}', AgentCategory.RETRIEVE),
        ('{"category": "recommend"}', AgentCategory.RECOMMEND),
        ('{"category": "observe"}', AgentCategory.OBSERVE),
    ],
)
def test_classify_query_parses_category(raw: str, expected: AgentCategory) -> None:
    context = make_context(llm=FakeLLM([raw]))
    state = classify_query("do something", context)
    assert state["category"] is expected
    assert any("Classified as" in log[1] for log in state["logs"])


def test_classify_query_missing_category_falls_back() -> None:
    context = make_context(llm=FakeLLM(['{"other_key": true}']))
    state = classify_query("anything", context)
    assert state["category"] is AgentCategory.RETRIEVE


def test_classify_query_unparseable_falls_back() -> None:
    context = make_context(llm=FakeLLM(["this is not json"]))
    state = classify_query("anything", context)
    assert state["category"] is AgentCategory.RETRIEVE


def test_classify_query_malformed_json_falls_back() -> None:
    context = make_context(llm=FakeLLM(['{"category": }']))
    state = classify_query("anything", context)
    assert state["category"] is AgentCategory.RETRIEVE


def test_classify_query_llm_error_falls_back() -> None:
    context = make_context(llm=FakeLLM([]))
    state = classify_query("anything", context)
    assert state["category"] is AgentCategory.RETRIEVE


def test_classify_intent_node_direct() -> None:
    node = build_nodes(make_context(llm=FakeLLM(['{"category": "act"}'])))["classify_intent"]
    update = node(initial_coordinator_state("do it"))
    assert update["category"] is AgentCategory.ACT
    assert update["logs"] == [("classify_intent", "Classified as 'act'")]


def test_run_agent_routes_to_retrieve() -> None:
    oracle = FakeOracle(
        fetch_schema=SAMPLE_SCHEMA,
        result=QueryResult(columns=["REGION"], rows=[["West", 100]]),
    )
    context = make_context(
        llm=FakeLLM(['{"category": "retrieve"}', GOOD_SQL, ANSWER]),
        connection=oracle,
    )
    state = run_agent("Sum sales by region", context)
    assert state["category"] is AgentCategory.RETRIEVE
    assert state["done"] is True
    assert state["error"] is None
    assert state["query_columns"] == ["REGION"]
    assert state["query_rows"] == [["West", 100]]
    assert state["sql_query"] == GOOD_SQL
    assert state["answer"] == ANSWER
    assert ("classify_intent", "Classified as 'retrieve'") in state["logs"]


def test_run_agent_routes_to_placeholder_agent() -> None:
    context = make_context(llm=FakeLLM(['{"category": "act"}']))
    state = run_agent("Create a listing", context)
    assert state["category"] is AgentCategory.ACT
    assert state["done"] is True
    assert "not yet implemented" in (state["error"] or "")
    assert any("not yet implemented" in log[1] for log in state["logs"])


def test_run_agent_merges_downstream_logs() -> None:
    goal = '{"goal_clear": true, "goal": "Grow Q3 revenue"}'
    data_needs = '{"data_needs": ["total revenue by region"]}'
    plan = (
        '{"understanding": "u", "data_used": "s", "key_findings": ["f"], '
        '"recommendations": ["r"], "data_gaps": "risk", "observe_duration": "30 days"}'
    )
    context = make_context(
        llm=FakeLLM(['{"category": "recommend"}', goal, data_needs, GOOD_SQL, plan]),
        connection=FakeOracle(
            fetch_schema=SAMPLE_SCHEMA,
            result=QueryResult(columns=["REGION"], rows=[["West", 100]]),
        ),
        input_fn=lambda _prompt: "approve",
    )
    state = run_agent("Suggest something", context)
    assert state["category"] is AgentCategory.RECOMMEND
    assert any(log[0] == "classify_intent" for log in state["logs"])
    assert any(log[0] == "recommend" for log in state["logs"])


def test_coordinator_main_modules_importable() -> None:
    import agents.__main__
    import agents.coordinator.__main__  # noqa: F401
