"""Shared fixtures for the SQL generator test suite."""

from collections.abc import Generator
import os

import pytest

from agents.common.config import get_settings


@pytest.fixture(autouse=True)
def _hermetic_environment() -> Generator[None, None, None]:
    """Isolate every test from real configuration and observability."""
    os.environ["LANGFUSE_ENABLED"] = "false"
    os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
    os.environ.pop("LANGFUSE_SECRET_KEY", None)
    os.environ.pop("LANGFUSE_HOST", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
