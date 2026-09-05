"""Rich-based output rendering shared by every agent."""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agents.common.state import BaseAgentState


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


def build_result_table(results: Results) -> Table:
    """Build a Rich table from ``results`` and its metadata caption."""
    table = Table(title="Query Results", show_lines=False)
    for column in results.columns:
        table.add_column(str(column), overflow="fold")
    for row in results.rows:
        table.add_row(*(str(cell) for cell in row))
    table.caption = _metadata_caption(results)
    return table


def print_agent_output(
    state: BaseAgentState,
    *,
    node_titles: Mapping[str, str] | None = None,
    results: Results | None = None,
    failure_message: str | None = None,
    console: Console | None = None,
) -> None:
    """Print the full agent output: per-node logs plus results (or error)."""
    output_console = console or Console()
    for panel in build_log_panels(state, node_titles):
        output_console.print(panel)
    if results is not None:
        output_console.print(build_result_table(results))
        return
    if failure_message is not None:
        output_console.print(build_failure_panel(failure_message))
        return
    output_console.print(Text("Agent finished without producing a result.", style="yellow"))


def _metadata_caption(results: Results) -> str:
    elapsed = results.execution_time_ms
    parts = [
        f"Executed in {elapsed:.1f} ms" if elapsed is not None else "No timing recorded",
        f"{results.attempt_count} generate/validate attempt(s)",
    ]
    if results.sql_query:
        shim = " ".join(results.sql_query.split())
        parts.append(f"SQL: {shim[:80]}{'...' if len(shim) > 80 else ''}")
    return " | ".join(parts)
