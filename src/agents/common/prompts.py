"""Prompt helpers shared across agents: tolerant JSON parsing from model output."""

import json
import re
from typing import Any, cast


def parse_json_object(raw: str) -> dict[str, Any]:
    """Tolerantly parse the first JSON object from a model response.

    Returns an empty dict when no object (or a malformed one) can be found.
    """
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return {}
    try:
        return cast("dict[str, Any]", json.loads(match.group(0)))
    except json.JSONDecodeError:
        return {}
