"""Langfuse observability integration for the SQL generator agent."""

from collections.abc import Callable
import functools
import os
from typing import Literal, ParamSpec, TypeVar

from dotenv import load_dotenv
from langfuse import observe as langfuse_observe

from lib.sql_generator.config import Settings, get_settings

_P = ParamSpec("_P")
_R = TypeVar("_R")
_AsType = Literal["span", "chain"]


def _ensure_env(settings: Settings) -> None:
    """Populate LANGFUSE_* environment variables from settings if absent.

    The langfuse SDK reads these directly from ``os.environ`` when it lazily
    initialises its client, so they must be present regardless of how settings
    were loaded.
    """
    load_dotenv()
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
    if settings.langfuse_enabled and settings.has_langfuse_credentials:
        os.environ["LANGFUSE_ENABLED"] = "true"
    else:
        os.environ["LANGFUSE_ENABLED"] = "false"


def _tracing_enabled(settings: Settings) -> bool:
    return bool(settings.langfuse_enabled and settings.has_langfuse_credentials)


def observe_step(name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Return a decorator that traces a graph step with Langfuse.

    Whether tracing is active is decided at call time, so decorating graph
    functions early has no side effects when observability is disabled.
    """
    return _trace_wrapper(name, as_type="span")


def observe_run(name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Return a decorator that traces the whole agent run as a root call."""
    return _trace_wrapper(name, as_type="chain")


def tracing_configured() -> bool:
    """Return True when Langfuse tracing is enabled and configured."""
    return _tracing_enabled(get_settings())


def _trace_wrapper(
    name: str,
    *,
    as_type: _AsType,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def _decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            settings = get_settings()
            if not _tracing_enabled(settings):
                return func(*args, **kwargs)
            _ensure_env(settings)
            observed = langfuse_observe(name=name, as_type=as_type)(func)
            return observed(*args, **kwargs)

        return _wrapper

    return _decorator
