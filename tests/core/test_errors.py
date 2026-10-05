"""Mistakes in a contest's files are reported as `OcimaticError`s."""

from __future__ import annotations

from pathlib import Path

import pytest

from ocimatic.core import load_contest
from ocimatic.errors import OcimaticError

from ..conftest import UseEnv
from ..contest import ABSENT, TaskSpec, make_contest
from ..text import block


def test_missing_statement(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum", statement=ABSENT))
    with (
        use_env(cwd=contest, contest_root=contest),
        pytest.raises(OcimaticError, match="statement file not found"),
    ):
        load_contest()


def test_invalid_task_config(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum", task_toml="task = [\n"))
    with (
        use_env(cwd=contest, contest_root=contest),
        pytest.raises(OcimaticError, match="Failed to load task config") as exc_info,
    ):
        load_contest()
    assert exc_info.value.details


def test_missing_testplan(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum", testplan=ABSENT))
    with (
        use_env(cwd=contest, contest_root=contest),
        pytest.raises(OcimaticError, match="File not found"),
    ):
        load_contest()


def test_all_testplan_errors_are_reported(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(
        tmp_path,
        TaskSpec(
            codename="sum",
            testplan=block("""
                [Subtask 1]
                  @extends subtask 0
                  a.b ; echo 1
            """),
        ),
    )
    with (
        use_env(cwd=contest, contest_root=contest),
        pytest.raises(OcimaticError, match="Error when parsing testplan") as exc_info,
    ):
        load_contest()
    details = exc_info.value.details
    assert details is not None
    assert "subtask number must be greater than or equal to 1" in details
    assert "invalid group name: `a.b`" in details
