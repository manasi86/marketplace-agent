"""Python date parsing and Oracle SQL literal formatting helpers."""

from datetime import date
import re
from typing import Any

_SEPARATOR = re.compile(r"\s*(?:to|through|until|\.\.)\s*", flags=re.IGNORECASE)


def parse_date_range(entities: dict[str, Any]) -> tuple[date | None, date | None]:
    """Parse the 'date_range' entity into (start, end) Python dates.

    Accepts single dates and ranges written as 'YYYY-MM-DD', with common
    separators like 'to', 'through', 'until' or '...'. Unparseable values
    yield ``None`` rather than raising.
    """
    raw = entities.get("date_range")
    if not raw:
        return (None, None)
    parts = [part.strip() for part in _SEPARATOR.split(str(raw)) if part.strip()]
    parsed = [d for part in parts if (d := _parse_iso(part)) is not None]
    if not parsed:
        return (None, None)
    if len(parsed) == 1:
        return (parsed[0], parsed[0])
    return (parsed[0], parsed[-1])


def _parse_iso(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def oracle_date_literal(d: date) -> str:
    """Format a Python date as an Oracle ``TO_DATE`` literal."""
    return f"TO_DATE('{d.isoformat()}', 'YYYY-MM-DD')"


def oracle_date_range(start: date | None, end: date | None) -> str | None:
    """Build an Oracle date-range clause, or ``None`` if no date is available."""
    if start is not None and end is not None:
        if start == end:
            return oracle_date_literal(start)
        return f"{oracle_date_literal(start)} TO {oracle_date_literal(end)}"
    if start is not None:
        return oracle_date_literal(start)
    if end is not None:
        return oracle_date_literal(end)
    return None
