"""Pretty CLI rendering with Rich for agent logs and query results."""

from collections import OrderedDict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lib.sql_generator.state import AgentState

NODE_TITLES = {
    "understand_intent": "1. Understand Intent",
    "check_db_connection": "2. Check DB Connection",
    "discover_semantic_layer": "3. Discover Semantic Layer",
    "generate_sql": "4. Generate SQL",
    "validate_sql": "5. Validate SQL",
    "execute_and_display": "6. Execute & Display",
}


def build_result_table(state: AgentState) -> Table:
    """Build a Rich table for query results and metadata."""
    columns = state.get("query_columns") or []
    rows = state.get("query_rows") or []
    table = Table(title="Query Results", show_lines=False)
    for column in columns:
        table.add_column(str(column), overflow="fold")
    for row in rows:
        table.add_row(*(str(cell) for cell in row))
    table.caption = _metadata_caption(state)
    return table


def build_log_panels(state: AgentState) -> list[Panel]:
    """Render each node's collected logs as its own panel."""
    grouped: OrderedDict[str, list[tuple[int, str]]] = OrderedDict()
    for index, (node_name, message) in enumerate(state.get("logs") or [], start=1):
        grouped.setdefault(node_name, []).append((index, message))
    panels: list[Panel] = []
    for node_name, entries in grouped.items():
        lines = [f"[dim]{index:>3}.[/dim] {message}" for index, message in entries]
        panels.append(
            Panel(
                "\n".join(lines),
                title=NODE_TITLES.get(node_name, node_name),
                border_style="cyan",
            )
        )
    return panels


def build_error_panel(state: AgentState) -> Panel:
    """Render the fatal error or final validation failure."""
    error = state.get("error")
    final_validation_error = state.get("validation_error")
    message = error or final_validation_error or "Unknown error."
    return Panel(Text(str(message), style="bold red"), title="Failure", border_style="red")


def print_agent_output(state: AgentState, console: Console | None = None) -> None:
    """Print the full agent output: per-node logs, results (or error)."""
    output_console = console or Console()
    for panel in build_log_panels(state):
        output_console.print(panel)
    if state.get("query_rows") is not None:
        output_console.print(build_result_table(state))
        return
    if state.get("error") is not None or state.get("validation_error") is not None:
        output_console.print(build_error_panel(state))
        return
    output_console.print(Text("Agent finished without producing a result.", style="yellow"))


def _metadata_caption(state: AgentState) -> str:
    elapsed = state.get("execution_time_ms")
    attempts = state.get("attempt_count") or 0
    parts = [
        f"Executed in {elapsed:.1f} ms" if elapsed is not None else "No timing recorded",
        f"{attempts} generate/validate attempt(s)",
    ]
    sql = state.get("sql_query")
    if sql:
        shim = " ".join(sql.split())
        parts.append(f"SQL: {shim[:80]}{'...' if len(shim) > 80 else ''}")
    return " | ".join(parts)
