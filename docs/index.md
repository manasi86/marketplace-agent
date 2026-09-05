---
icon: material/math-integral
status: new
---

# `marketplace-Agent` User Guide

!!! info

    This user guide is purely an illustrative example that shows off several features of
    [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) and included Markdown
    extensions[^1].

[^1]: See `marketplace-Agent`'s `mkdocs.yml` for how to enable these features.

## Installation

First, [install `uv`](https://docs.astral.sh/uv/getting-started/installation):

=== "macOS and Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

Then install the project and its dependencies:

```bash
uv sync
```

## Quick Start

To use `lib` as a library within your project, import it and execute the API like:

*[API]: Application Programming Interface

```python
import lib
import api
```

!!! tip

    Within PyCharm, use ++tab++ to auto-complete suggested imports while typing.