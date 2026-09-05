"""Enable ``python -m agents`` (runs the coordinator CLI)."""

from agents.coordinator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
