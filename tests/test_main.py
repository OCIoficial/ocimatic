"""Tests for the commands in `ocimatic.main`.

Commands are tested through the `Status` they return, not their exit code: `run_command` parses
the arguments like the CLI would and calls the command without its exit-code decorator. The
environment the `cli` group would install comes from the `use_env` fixture.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import click
import pytest

from ocimatic.main import cli
from ocimatic.result import Status

from .conftest import UseEnv
from .contest import TaskSpec, make_contest
from .text import block


def run_command(name: str, *args: str) -> Status:
    command = cli.commands[name]
    assert command.callback is not None
    ctx = command.make_context(name, list(args))
    with ctx:
        result = ctx.invoke(inspect.unwrap(command.callback), **ctx.params)
    assert isinstance(result, Status)
    return result


FAILING_GENERATOR = TaskSpec(
    codename="sum",
    testplan=block("""
        [Subtask 1]
          rand ; gen.py
    """),
    files={"testplan/gen.py": "raise SystemExit(1)\n"},
)


def test_run_testplan_succeeds(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    with use_env(cwd=contest / "sum", contest_root=contest):
        assert run_command("run-testplan") == Status.success


def test_run_testplan_fails_on_single_task(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(tmp_path, FAILING_GENERATOR)
    with use_env(cwd=contest / "sum", contest_root=contest):
        assert run_command("run-testplan") == Status.fail


def test_run_testplan_fails_on_some_task(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="ok"), FAILING_GENERATOR)
    with use_env(cwd=contest, contest_root=contest):
        assert run_command("run-testplan") == Status.fail


def test_gen_expected_succeeds(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    with use_env(cwd=contest / "sum", contest_root=contest):
        assert run_command("run-testplan") == Status.success
        assert run_command("gen-expected") == Status.success


def test_gen_expected_fails_on_single_task(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(
        tmp_path,
        TaskSpec(codename="sum", correct={"crash.py": "raise SystemExit(1)\n"}),
    )
    with use_env(cwd=contest / "sum", contest_root=contest):
        assert run_command("run-testplan") == Status.success
        assert run_command("gen-expected") == Status.fail


def test_run_fails_when_solution_is_missing(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    with use_env(cwd=contest / "sum", contest_root=contest):
        assert run_command("run", "missing.py") == Status.fail


def test_build_fails_when_solution_is_missing(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    with use_env(cwd=contest / "sum", contest_root=contest):
        assert run_command("build", "missing.py") == Status.fail


def test_score_params_fails_on_subtask_mismatch(
    tmp_path: Path,
    use_env: UseEnv,
) -> None:
    contest = make_contest(
        tmp_path,
        TaskSpec(
            codename="sum",
            statement=block("""
                #subtask(40)
                #subtask(60)
            """),
        ),
    )
    with use_env(cwd=contest / "sum", contest_root=contest):
        assert run_command("score-params") == Status.fail


@pytest.mark.parametrize(
    "args",
    [
        ["run", "sum.py", "--subtask"],
        ["run-testplan", "--subtask"],
        ["validate-input", "--subtask"],
        ["validate-output", "--subtask"],
    ],
    ids=lambda args: args[0],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_subtask_must_be_positive(args: list[str], value: str) -> None:
    name, *rest = args
    with pytest.raises(click.BadParameter, match="x>=1"):
        cli.commands[name].make_context(name, [*rest, value])
