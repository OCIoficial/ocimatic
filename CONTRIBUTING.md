# Contributing to Ocimatic

## Development setup

Ocimatic uses [uv](https://docs.astral.sh/uv/) for development. To set up the environment, run:

```bash
uv sync
```

This creates a `.venv` with ocimatic installed in editable mode, so `uv run ocimatic` runs your
local checkout.

## Tests

Tests live in `tests/` and use [pytest](https://docs.pytest.org/).

```bash
uv run pytest           # unit tests
uv run pytest -m e2e    # end-to-end tests
```

End-to-end tests need a full environment including `javac` and a C++ compiler
(`clang++` by default, configurable with `ocimatic setup`). They are excluded from
the default run.

## Linting

CI also runs the following checks:

```bash
uv run mypy .
uv run pyright .
uv run ruff check
uv run ruff format --check
```

## Releasing the VS Code extension

The extension lives in `tools/ocimatic-testplan` and is published to the
[VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=nlehmann.ocimatic-testplan)
under the `nlehmann` publisher. New versions are uploaded by hand.

1. Bump `version` in `tools/ocimatic-testplan/package.json`.
2. Add a `## [<version>]` entry to `tools/ocimatic-testplan/CHANGELOG.md`.
3. Build the `.vsix` package (requires Node.js 22 or newer):

   ```bash
   cd tools/ocimatic-testplan
   npm ci
   npm run package
   ```

   This creates `ocimatic-testplan-<version>.vsix`.
4. Go to the [Marketplace publisher page](https://marketplace.visualstudio.com/manage/publishers/nlehmann)
   and sign in with the Microsoft account that owns the publisher.
5. Open the **⋯** menu next to *ocimatic-testplan*, select **Update**, and upload the `.vsix`.

The Marketplace verifies the package, and the new version shows up after a few minutes.
