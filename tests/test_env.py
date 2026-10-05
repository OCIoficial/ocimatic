from __future__ import annotations

import os
from pathlib import Path

import pytest

from ocimatic.config import Config
from ocimatic.env import Env, Verbosity
from ocimatic.errors import OcimaticError
from ocimatic.utils import relative_to_cwd


def _env(cwd: Path, contest_root: Path | None) -> Env:
    return Env(config=Config(), cwd=cwd, contest_root=contest_root)


def test_get_without_env_raises() -> None:
    with pytest.raises(RuntimeError, match="no environment installed"):
        Env.get()


def test_use_restores_previous_env(tmp_path: Path) -> None:
    outer = _env(tmp_path / "outer", None)
    inner = _env(tmp_path / "inner", None)
    with Env.use(outer):
        with Env.use(inner):
            assert Env.get() is inner
        assert Env.get() is outer
    with pytest.raises(RuntimeError):
        Env.get()


def test_override_changes_fields_and_restores(tmp_path: Path) -> None:
    env = _env(tmp_path, None)
    with Env.use(env):
        with Env.override(verbosity=Verbosity.quiet) as overridden:
            assert Env.get().verbosity is Verbosity.quiet
            assert overridden.cwd == env.cwd
        assert Env.get() is env


def test_require_contest_root_outside_contest(tmp_path: Path) -> None:
    with pytest.raises(OcimaticError, match="not called inside a contest"):
        _env(tmp_path, None).require_contest_root()


def test_paths_inside_contest_are_relative(tmp_path: Path) -> None:
    contest = tmp_path / "contest"
    with Env.use(_env(contest / "task", contest)):
        assert relative_to_cwd(contest / "task" / "sol.py") == f".{os.sep}sol.py"
        assert relative_to_cwd(contest / "other" / "sol.py") == str(
            Path("..", "other", "sol.py"),
        )


def test_paths_outside_contest_are_absolute(tmp_path: Path) -> None:
    contest = tmp_path / "contest"
    outside = tmp_path / "elsewhere" / "sol.py"
    with Env.use(_env(contest / "task", contest)):
        assert relative_to_cwd(outside) == str(outside)
