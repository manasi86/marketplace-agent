"""Semantic layer: database schema discovery plus prompt-friendly formatting."""

from typing import Any

from agents.common.db import OracleConnection

_MAX_FORMATTED_CHARS = 20_000


class SemanticContext:
    """Cached, schema-aware view of the Oracle data dictionary."""

    def __init__(self) -> None:
        """Initialise with no metadata loaded yet."""
        self._metadata: dict[str, Any] | None = None

    def discover(self, connection: OracleConnection) -> dict[str, Any]:
        """Load the semantic metadata, caching it across calls."""
        if self._metadata is None:
            self._metadata = connection.fetch_schema()
        return self._metadata

    @property
    def metadata(self) -> dict[str, Any]:
        """Return cached metadata, raising if discovery has not run yet."""
        if self._metadata is None:
            raise RuntimeError("Semantic discovery has not been run yet.")
        return self._metadata

    def format_for_prompt(self, schema_hint: str | None = None) -> str:
        """Render a compact textual summary of relevant tables and columns.

        When a schema hint is supplied, only that schema is included; otherwise
        every discovered schema is rendered.
        """
        metadata = self.metadata
        schemas = self._select_schemas(metadata, schema_hint)
        if not schemas:
            return "No tables or views found in the database."
        lines: list[str] = []
        for schema_name in sorted(schemas):
            tables = schemas[schema_name].get("tables", {})
            if not tables:
                continue
            lines.append(f"Schema: {schema_name}")
            for table_name in sorted(tables):
                table = tables[table_name]
                description = table.get("description")
                lines.append(f"  {table['type']} {table_name}: {description or 'no comment'}")
                for column_name in sorted(table.get("columns", {})):
                    column = table["columns"][column_name]
                    lines.append(
                        f"    - {column_name} ({column['type']}): "
                        f"{column.get('description') or 'no comment'}"
                    )
        rendered = "\n".join(lines)
        if len(rendered) > _MAX_FORMATTED_CHARS:
            rendered = rendered[:_MAX_FORMATTED_CHARS] + (
                f"\n... (semantic layer truncated at {_MAX_FORMATTED_CHARS} chars; "
                "provide a schema hint to narrow it)"
            )
        return rendered

    def _select_schemas(
        self,
        metadata: dict[str, Any],
        schema_hint: str | None,
    ) -> dict[str, Any]:
        if not schema_hint:
            return metadata
        hint = schema_hint.strip().upper()
        matches = {name: data for name, data in metadata.items() if hint in name.upper()}
        return matches or metadata
