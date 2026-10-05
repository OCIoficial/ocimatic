from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from ocimatic.config import Config
from ocimatic.env import Env

type UseEnv = Callable[..., AbstractContextManager[Env]]


@pytest.fixture(autouse=True)
def _isolate_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read the user's `~/.ocimatic.toml`."""
    monkeypatch.setattr(Config, "HOME_PATH", tmp_path / "home" / ".ocimatic.toml")


@pytest.fixture
def use_env() -> UseEnv:
    """Return a function that installs an environment with the default configuration."""

    def _use(
        *,
        cwd: Path,
        contest_root: Path | None = None,
    ) -> AbstractContextManager[Env]:
        return Env.use(Env(config=Config(), cwd=cwd, contest_root=contest_root))

    return _use
