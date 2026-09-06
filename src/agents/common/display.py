"""Rich-based output rendering shared by every agent."""

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agents.common.config import get_settings
from agents.common.state import BaseAgentState

DEFAULT_PAGE_SIZE = 50


@dataclass(frozen=True)
class Results:
    """Optional tabular results rendered as a Rich table."""

    columns: Sequence[str] = field(default_factory=list)
    rows: Sequence[Sequence[Any]] = field(default_factory=list)
    execution_time_ms: float | None = None
    sql_query: str = ""
    attempt_count: int = 0


def build_log_panels(
    state: BaseAgentState,
    node_titles: Mapping[str, str] | None = None,
) -> list[Panel]:
    """Render each node's collected logs as its own panel."""
    grouped: OrderedDict[str, list[tuple[int, str]]] = OrderedDict()
    titles = node_titles or {}
    for index, (node_name, message) in enumerate(state.get("logs") or [], start=1):
        grouped.setdefault(node_name, []).append((index, message))
    panels: list[Panel] = []
    for node_name, entries in grouped.items():
        lines = [f"[dim]{index:>3}.[/dim] {message}" for index, message in entries]
        panels.append(
            Panel(
                "\n".join(lines),
                title=titles.get(node_name, node_name),
                border_style="cyan",
            )
        )
    return panels


def build_failure_panel(message: str) -> Panel:
    """Render a fatal error or final validation failure."""
    return Panel(Text(message, style="bold red"), title="Failure", border_style="red")


def build_answer_panel(answer: str) -> Panel:
    """Render an agent's natural-language answer prominently."""
    return Panel(Text(answer.strip(), style="bold green"), title="Answer", border_style="green")


def build_result_table(
    results: Results,
    *,
    page_start: int | None = None,
    total_rows: int | None = None,
) -> Table:
    """Build a Rich table from ``results`` and its metadata caption.

    ``page_start`` (1-based) and ``total_rows`` allow a page footer such as
    "showing rows 1-50 of 120" to be appended to the caption.
    """
    table = Table(title="Query Results", show_lines=False)
    for column in results.columns:
        table.add_column(str(column), overflow="fold")
    for row in results.rows:
        table.add_row(*(str(cell) for cell in row))
    table.caption = _metadata_caption(results, page_start=page_start, total_rows=total_rows)
    return table


def print_agent_output(
    state: BaseAgentState,
    *,
    node_titles: Mapping[str, str] | None = None,
    results: Results | None = None,
    failure_message: str | None = None,
    answer: str | None = None,
    console: Console | None = None,
) -> None:
    """Print the full agent output: per-node logs plus answer and results (or error)."""
    output_console = console or Console()
    for panel in build_log_panels(state, node_titles):
        output_console.print(panel)
    if answer is not None:
        output_console.print(build_answer_panel(answer))
    if results is not None:
        print_result_table_paginated(results, output_console)
        return
    if failure_message is not None:
        output_console.print(build_failure_panel(failure_message))
        return
    output_console.print(Text("Agent finished without producing a result.", style="yellow"))


def _page_size() -> int:
    """Return the configured results page size, falling back to the default."""
    try:
        return get_settings().results_page_size
    except Exception:  # pragma: no cover - defensive if settings are unavailable
        return DEFAULT_PAGE_SIZE


def _is_interactive() -> bool:
    """Return True when standard input is an interactive terminal."""
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except Exception:  # pragma: no cover - defensive across environments
        return False


def _render_page(
    results: Results,
    start: int,
    page_size: int,
    console: Console,
) -> None:
    """Render one 1-based ``start`` page of ``results`` to ``console``."""
    page_rows = results.rows[start - 1 : start - 1 + page_size]
    page_results = Results(
        columns=results.columns,
        rows=page_rows,
        execution_time_ms=results.execution_time_ms,
        sql_query=results.sql_query,
        attempt_count=results.attempt_count,
    )
    console.print(
        build_result_table(
            page_results,
            page_start=start,
            total_rows=len(results.rows),
        )
    )


def _resolve_page(page: int, pages: int) -> int | None:
    """Clamp a 1-based page number into range, or None when out of range."""
    if 1 <= page <= pages:
        return page
    return None


def _run_pager(
    results: Results,
    console: Console,
    page_size: int,
    *,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Interactively page through a large result table.

    Commands: n (next), p (previous), j <page> (jump), q (quit). Returns when
    the user quits or input is exhausted.
    """
    total = len(results.rows)
    pages = max(1, (total + page_size - 1) // page_size)
    page = 1

    while True:
        start = (page - 1) * page_size + 1
        _render_page(results, start, page_size, console)
        prompt = (
            f"Page {page}/{pages} "
            f"[{'-'.join(map(str, ((page - 1) * page_size + 1, min(page * page_size, total))))}] "
            "\u2014 [n]ext, [p]rev, [j]ump <page>, [q]uit > "
        )
        try:
            raw = (input_fn(prompt) or "").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if raw in ("q", "quit", ""):
            return
        if raw in ("n", "next"):
            page = page + 1 if page < pages else page
        elif raw in ("p", "prev", "previous"):
            page = page - 1 if page > 1 else page
        elif raw.startswith("j"):
            parts = raw.split()
            if len(parts) == 2 and parts[1].isdigit():
                target = _resolve_page(int(parts[1]), pages)
                if target is not None:
                    page = target
        console.clear()


def print_result_table_paginated(
    results: Results,
    console: Console | None = None,
    *,
    page_size: int | None = None,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Render ``results`` as a Rich table, paging interactively when it is large.

    When the result has more rows than ``page_size`` (default 50) it is shown
    one page at a time. Interactive paging only activates when stdin is a TTY;
    otherwise only the first page is printed and the function returns.
    """
    output_console = console or Console()
    resolved_page_size = page_size or _page_size()
    total = len(results.rows)

    if total <= resolved_page_size or not _is_interactive():
        _render_page(results, 1, resolved_page_size, output_console)
        return

    _run_pager(results, output_console, resolved_page_size, input_fn=input_fn)


def _metadata_caption(
    results: Results,
    *,
    page_start: int | None = None,
    total_rows: int | None = None,
) -> str:
    elapsed = results.execution_time_ms
    parts = [
        f"Executed in {elapsed:.1f} ms" if elapsed is not None else "No timing recorded",
        f"{results.attempt_count} generate/validate attempt(s)",
    ]
    if results.sql_query:
        shim = " ".join(results.sql_query.split())
        parts.append(f"SQL: {shim[:80]}{'...' if len(shim) > 80 else ''}")
    if page_start is not None and total_rows is not None:
        page_end = min(page_start + len(results.rows) - 1, total_rows)
        parts.append(f"showing rows {page_start}-{page_end} of {total_rows}")
    return " | ".join(parts)
