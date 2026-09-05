"""Enable ``python -m agents.retrieve`` (runs the Retrieve agent CLI)."""

from agents.retrieve.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
