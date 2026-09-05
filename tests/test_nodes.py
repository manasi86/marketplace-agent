"""Tests for the graph node functions."""

from datetime import date
from typing import Any, cast

from agents.common.config import Settings
from agents.common.db import QueryResult
from agents.common.semantic import SemanticContext
from agents.sql_generator.nodes import _parse_intent_json, build_nodes
from agents.sql_generator.state import SqlGeneratorState, initial_state
from tests.doubles import FakeLLM, FakeOracle, make_context

INTENT_JSON = (
    '{"intent": "aggregate sales", "entities": {"region": "West"}, "schema_hint": "SALES"}'
)
GOOD_SQL = "SELECT region, SUM(total) FROM SALES.VW_SALES_SUMMARY GROUP BY region"

SAMPLE_SCHEMA: dict[str, Any] = {
    "SALES": {
        "tables": {
            "VW_SALES_SUMMARY": {
                "type": "VIEW",
                "description": "Sales",
                "columns": {
                    "REGION": {"type": "VARCHAR2", "description": "Region"},
                    "TOTAL": {"type": "NUMBER", "description": "Total"},
                },
            }
        }
    }
}


def _node_state(**overrides: Any) -> SqlGeneratorState:
    state = initial_state("Show sales", 3)
    state.update(cast(Any, overrides))
    return state


def test_understand_intent_parses_json() -> None:
    context = make_context(llm=FakeLLM([INTENT_JSON]))
    node = build_nodes(context)["understand_intent"]
    update = node(_node_state(user_query="Show sales by region"))
    assert update["intent"] == "aggregate sales"
    assert update["entities"] == {"region": "West"}
    assert update["schema_hint"] == "SALES"
    assert ("understand_intent", "Understood intent: aggregate sales") in update["logs"]


def test_understand_intent_with_null_schema_hint() -> None:
    llm = FakeLLM(['{"intent": "count rows", "entities": {}, "schema_hint": null}'])
    context = make_context(llm=llm)
    node = build_nodes(context)["understand_intent"]
    update = node(_node_state(user_query="How many rows?"))
    assert update["schema_hint"] is None
    assert update["entities"] == {}


def test_understand_intent_parses_date_range() -> None:
    llm = FakeLLM(
        [
            (
                '{"intent": "sales by day", '
                '"entities": {"date_range": "2025-01-01 to 2025-12-31"}, '
                '"schema_hint": null}'
            )
        ]
    )
    context = make_context(llm=llm)
    node = build_nodes(context)["understand_intent"]
    update = node(_node_state(user_query="Sales in 2025"))
    assert update["date_start"] == date(2025, 1, 1)
    assert update["date_end"] == date(2025, 12, 31)


def test_understand_intent_ignores_bad_date_range() -> None:
    llm = FakeLLM(
        ['{"intent": "x", "entities": {"date_range": "never ever"}, "schema_hint": null}']
    )
    context = make_context(llm=llm)
    node = build_nodes(context)["understand_intent"]
    update = node(_node_state(user_query="what up"))
    assert update["date_start"] is None
    assert update["date_end"] is None


def test_understand_intent_llm_error_falls_back() -> None:
    context = make_context(llm=FakeLLM([]))
    node = build_nodes(context)["understand_intent"]
    update = node(_node_state(user_query="count everything"))
    assert update["intent"] == "count everything"
    assert update["entities"] == {}
    assert update["schema_hint"] is None


def test_understand_intent_unparseable_falls_back() -> None:
    context = make_context(llm=FakeLLM(["this is not json"]))
    node = build_nodes(context)["understand_intent"]
    update = node(_node_state(user_query="list orders"))
    assert update["intent"] == "list orders"


def test_check_db_connection_success() -> None:
    context = make_context(connection=FakeOracle(connected=True))
    node = build_nodes(context)["check_db_connection"]
    update = node(_node_state())
    assert update["db_connected"] is True
    assert update["logs"] == [("check_db_connection", "Database connection check: OK")]


def test_check_db_connection_failure() -> None:
    context = make_context(connection=FakeOracle(connected=False))
    node = build_nodes(context)["check_db_connection"]
    update = node(_node_state())
    assert update["db_connected"] is False
    assert update["done"] is True
    assert update["error"] == "Could not connect to the Oracle database."


def test_discover_semantic_layer_no_hint() -> None:
    semantic = SemanticContext()
    semantic._metadata = SAMPLE_SCHEMA
    context = make_context(connection=FakeOracle(), semantic=semantic)
    node = build_nodes(context)["discover_semantic_layer"]
    update = node(_node_state(schema_hint=None))
    assert update["semantic_metadata"] == SAMPLE_SCHEMA
    assert "Schema: SALES" in update["semantic_context"]
    assert "1 schema(s)" in update["logs"][0][1]


def test_discover_semantic_layer_with_hint() -> None:
    semantic = SemanticContext()
    semantic._metadata = SAMPLE_SCHEMA
    semantic_ref = semantic
    context = make_context(semantic=semantic_ref)
    node = build_nodes(context)["discover_semantic_layer"]
    update = node(_node_state(schema_hint="SALES"))
    assert "(schema hint: SALES)" in update["logs"][0][1]


def test_generate_sql_without_validation_error() -> None:
    context = make_context(llm=FakeLLM([GOOD_SQL]))
    node = build_nodes(context)["generate_sql"]
    update = node(
        _node_state(
            intent="aggregate sales",
            entities={"region": "West"},
            semantic_context="Schema: SALES",
            validation_error=None,
        )
    )
    assert update["sql_query"] == GOOD_SQL
    assert "(fix attempt" not in update["logs"][0][1]


def test_generate_sql_with_validation_error() -> None:
    context = make_context(llm=FakeLLM([GOOD_SQL]))
    node = build_nodes(context)["generate_sql"]
    update = node(
        _node_state(
            intent="aggregate sales",
            entities={},
            semantic_context="Schema: SALES",
            validation_error="ORA-00904 invalid identifier",
        )
    )
    assert update["sql_query"] == GOOD_SQL
    assert "(fix attempt after validation error)" in update["logs"][0][1]


def test_generate_sql_llm_failure() -> None:
    context = make_context(llm=FakeLLM([]))
    node = build_nodes(context)["generate_sql"]
    update = node(_node_state(intent="x", entities={}, semantic_context="Schema: SALES"))
    assert update["error"] == "No scripted responses left."
    assert update["done"] is True


def test_validate_sql_success() -> None:
    context = make_context(connection=FakeOracle(explain_failures=0))
    node = build_nodes(context)["validate_sql"]
    update = node(_node_state(sql_query=GOOD_SQL, attempt_count=0, max_attempts=3))
    assert update["validation_error"] is None
    assert "PASSED" in update["logs"][0][1]


def test_validate_sql_failure_below_max() -> None:
    context = make_context(connection=FakeOracle(explain_failures=10))
    node = build_nodes(context)["validate_sql"]
    update = node(_node_state(sql_query=GOOD_SQL, attempt_count=0, max_attempts=3))
    assert update["attempt_count"] == 1
    assert "invalid identifier" in (update["validation_error"] or "")
    assert "done" not in update or update["done"] is False


def test_validate_sql_failure_exhausts_max() -> None:
    context = make_context(connection=FakeOracle(explain_failures=10))
    node = build_nodes(context)["validate_sql"]
    update = node(_node_state(sql_query=GOOD_SQL, attempt_count=2, max_attempts=3))
    assert update["attempt_count"] == 3
    assert "after 3 attempts" in (update["error"] or "")
    assert update["done"] is True
    assert "FAILED (attempt 3/3)" in update["logs"][0][1]


def test_execute_and_display_success() -> None:
    context = make_context(
        connection=FakeOracle(result=QueryResult(columns=["REGION"], rows=[["West", 100]])),
    )
    node = build_nodes(context)["execute_and_display"]
    update = node(_node_state(sql_query=GOOD_SQL))
    assert update["query_columns"] == ["REGION"]
    assert update["query_rows"] == [["West", 100]]
    assert update["done"] is True
    assert isinstance(update["execution_time_ms"], float)
    assert "1 row(s)" in update["logs"][0][1]


def test_execute_and_display_database_error() -> None:
    context = make_context(
        connection=FakeOracle(execute_error="ORA-00942: table or view does not exist"),
    )
    node = build_nodes(context)["execute_and_display"]
    update = node(_node_state(sql_query=GOOD_SQL))
    assert update["error"] == "ORA-00942: table or view does not exist"
    assert update["done"] is True


def test_parse_intent_json_success() -> None:
    assert _parse_intent_json(INTENT_JSON)["intent"] == "aggregate sales"


def test_parse_intent_json_no_json_match() -> None:
    assert _parse_intent_json("nothing here") == {
        "intent": "",
        "entities": {},
        "schema_hint": None,
    }


def test_parse_intent_json_malformed() -> None:
    assert _parse_intent_json('{"intent": }') == {
        "intent": "",
        "entities": {},
        "schema_hint": None,
    }


def test_settings_default_in_context() -> None:
    settings = Settings(sql_gen_api_key="k", langfuse_enabled=False)
    context = make_context(settings=settings)
    assert context.settings.max_sql_attempts == 3
