"""Tests for the date parsing and Oracle formatting helpers."""

from datetime import date

from lib.sql_generator.dates import oracle_date_literal, oracle_date_range, parse_date_range


def test_parse_date_range_full_range() -> None:
    assert parse_date_range({"date_range": "2025-01-01 to 2025-12-31"}) == (
        date(2025, 1, 1),
        date(2025, 12, 31),
    )


def test_parse_date_range_missing() -> None:
    assert parse_date_range({}) == (None, None)
    assert parse_date_range({"date_range": None}) == (None, None)


def test_parse_date_range_single_date() -> None:
    assert parse_date_range({"date_range": "2025-06-01"}) == (
        date(2025, 6, 1),
        date(2025, 6, 1),
    )


def test_parse_date_range_alt_separators() -> None:
    assert parse_date_range({"date_range": "2025-01-01 through 2025-02-01"}) == (
        date(2025, 1, 1),
        date(2025, 2, 1),
    )
    assert parse_date_range({"date_range": "2025-01-01..2025-02-01"}) == (
        date(2025, 1, 1),
        date(2025, 2, 1),
    )


def test_parse_date_range_keeps_iso_hyphens() -> None:
    assert parse_date_range({"date_range": "2025-01-01 to 2025-12-31"}) == (
        date(2025, 1, 1),
        date(2025, 12, 31),
    )


def test_parse_date_range_invalid_returns_none() -> None:
    assert parse_date_range({"date_range": "never ever"}) == (None, None)
    assert parse_date_range({"date_range": ""}) == (None, None)


def test_oracle_date_literal() -> None:
    assert oracle_date_literal(date(2025, 1, 1)) == "TO_DATE('2025-01-01', 'YYYY-MM-DD')"


def test_oracle_date_range() -> None:
    result = oracle_date_range(date(2025, 1, 1), date(2025, 12, 31))
    assert result == ("TO_DATE('2025-01-01', 'YYYY-MM-DD') TO TO_DATE('2025-12-31', 'YYYY-MM-DD')")


def test_oracle_date_range_none() -> None:
    assert oracle_date_range(None, None) is None


def test_oracle_date_range_single_or_missing_side() -> None:
    assert oracle_date_range(date(2025, 6, 1), date(2025, 6, 1)) == (
        "TO_DATE('2025-06-01', 'YYYY-MM-DD')"
    )
    assert oracle_date_range(None, date(2025, 6, 1)) == ("TO_DATE('2025-06-01', 'YYYY-MM-DD')")
    assert oracle_date_range(date(2025, 6, 1), None) == ("TO_DATE('2025-06-01', 'YYYY-MM-DD')")
