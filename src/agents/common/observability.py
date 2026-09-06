"""Langfuse observability integration shared by every agent."""

from collections.abc import Callable
from contextlib import nullcontext
import functools
import logging
import os
from time import perf_counter
from typing import Any, Literal, ParamSpec, TypeVar

from dotenv import load_dotenv
from langfuse import get_client as langfuse_get_client
from langfuse import observe as langfuse_observe

from agents.common.config import Settings, get_settings

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


def _get_logger(func: Callable[_P, _R]) -> logging.Logger:
    """Return the logger for the module that defined ``func``."""
    return logging.getLogger(func.__module__)


def observe_step(name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Return a decorator that traces a graph step with Langfuse.

    Whether tracing is active is decided at call time, so decorating graph
    functions early has no side effects when observability is disabled.
    """
    return _trace_wrapper(name, as_type="span")


def observe_run(name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Return a decorator that traces the whole agent run as a root call."""
    return _trace_wrapper(name, as_type="chain")


class _NoopObservation:
    """Absorbs observation updates when Langfuse tracing is disabled."""

    def update(self, **_: Any) -> None:
        return None


def observe_generation(name: str, input_value: Any) -> Any:
    """Return a Langfuse generation context manager for an LLM call.

    When tracing is disabled this returns a no-op context whose ``update``
    calls are absorbed, so callers do not need to branch. When enabled the
    generation nests under the current span/chain and carries the input.
    """
    settings = get_settings()
    if not _tracing_enabled(settings):
        return nullcontext(_NoopObservation())
    _ensure_env(settings)
    return langfuse_get_client().start_as_current_observation(
        name=name,
        as_type="generation",
        input=input_value,
    )


def tracing_configured() -> bool:
    """Return True when Langfuse tracing is enabled and configured."""
    return _tracing_enabled(get_settings())


def log_step(name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Return a decorator that logs each graph step with its duration."""

    def _decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            logger = _get_logger(func)
            logger.info("Step [%s] started", name)
            start = perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception("Step [%s] failed", name)
                raise
            elapsed_ms = (perf_counter() - start) * 1000.0
            logger.info("Step [%s] completed in %.1f ms", name, elapsed_ms)
            return result

        return _wrapped

    return _decorator


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
