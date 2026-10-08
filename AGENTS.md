# Agent Development Guide

A file for [guiding coding agents](https://agents.md/).

## Commands

Run all tools through `uv run`.

- **Test:** `uv run pytest`
- **Test (e2e):** `uv run pytest -m e2e`
- **Type check (mypy):** `uv run mypy .`
- **Type check (pyright):** `uv run pyright .`
- **Lint:** `uv run ruff check`
- **Formatting:** `uv run ruff format`
