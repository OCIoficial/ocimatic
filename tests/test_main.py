"""Tests for the commands in `ocimatic.main`.

Commands are run through the CLI with click's `CliRunner` and tested through their exit code:
0 for success, 1 for an `OcimaticError`, and 2 for a failed `Status` or invalid arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from ocimatic.main import cli

from .contest import ABSENT, TaskSpec, make_contest
from .text import block

type RunCli = Callable[..., Result]


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch) -> RunCli:
    """Return a function that runs `ocimatic <args>` from the directory `cwd`."""

    def _run(*args: str, cwd: Path) -> Result:
        monkeypatch.chdir(cwd)
        return CliRunner().invoke(cli, list(args))

    return _run


FAILING_GENERATOR = TaskSpec(
    codename="sum",
    testplan=block("""
        [Subtask 1]
          rand ; gen.py
    """),
    files={"testplan/gen.py": "raise SystemExit(1)\n"},
)


def test_run_testplan_succeeds(tmp_path: Path, run_cli: RunCli) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    assert run_cli("run-testplan", cwd=contest / "sum").exit_code == 0


def test_run_testplan_reports_validation_errors(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    contest = make_contest(
        tmp_path,
        TaskSpec(
            codename="sum",
            testplan=block("""
                [Subtask 2]
                  small ; echo 1 2
            """),
        ),
    )
    result = run_cli("run-testplan", cwd=contest / "sum")
    assert result.exit_code == 1
    assert "found [Subtask 2], but [Subtask 1] was expected" in result.output


def test_run_testplan_fails_on_single_task(tmp_path: Path, run_cli: RunCli) -> None:
    contest = make_contest(tmp_path, FAILING_GENERATOR)
    assert run_cli("run-testplan", cwd=contest / "sum").exit_code == 2


def test_run_testplan_fails_on_some_task(tmp_path: Path, run_cli: RunCli) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="ok"), FAILING_GENERATOR)
    assert run_cli("run-testplan", cwd=contest).exit_code == 2


def test_gen_expected_succeeds(tmp_path: Path, run_cli: RunCli) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    assert run_cli("run-testplan", cwd=contest / "sum").exit_code == 0
    assert run_cli("gen-expected", cwd=contest / "sum").exit_code == 0


def test_gen_expected_fails_on_single_task(tmp_path: Path, run_cli: RunCli) -> None:
    contest = make_contest(
        tmp_path,
        TaskSpec(
            codename="sum",
            solutions={"correct/crash.py": "raise SystemExit(1)\n"},
        ),
    )
    assert run_cli("run-testplan", cwd=contest / "sum").exit_code == 0
    assert run_cli("gen-expected", cwd=contest / "sum").exit_code == 2


def test_run_fails_when_solution_is_missing(tmp_path: Path, run_cli: RunCli) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    assert run_cli("run", "missing.py", cwd=contest / "sum").exit_code == 2


def test_build_fails_when_solution_is_missing(tmp_path: Path, run_cli: RunCli) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    assert run_cli("build", "missing.py", cwd=contest / "sum").exit_code == 2


def test_score_params_fails_on_subtask_mismatch(
    tmp_path: Path,
    run_cli: RunCli,
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
    assert run_cli("score-params", cwd=contest / "sum").exit_code == 2


def test_score_params_succeeds(tmp_path: Path, run_cli: RunCli) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    assert run_cli("run-testplan", cwd=contest / "sum").exit_code == 0
    assert run_cli("score-params", cwd=contest / "sum").exit_code == 0


def test_score_params_fails_on_subtask_without_tests(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    # The testplan hasn't been run, so the task's single subtask has no tests.
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    result = run_cli("score-params", cwd=contest / "sum")
    assert result.exit_code == 2, result.output
    assert "subtasks without tests: 1." in result.output


def test_score_params_fails_on_some_subtask_without_tests(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    contest = make_contest(
        tmp_path,
        TaskSpec(
            codename="sum",
            static=True,
            testplan=ABSENT,
            statement=block("""
                #let subtask(points) = [Subtask (#points points)]
                #subtask(40)
                #subtask(60)
            """),
            dataset={"st1": {"a.in": "1 2\n", "a.sol": "3\n"}, "st2": {}},
        ),
    )
    result = run_cli("score-params", cwd=contest / "sum")
    assert result.exit_code == 2, result.output
    assert "subtasks without tests: 2." in result.output


def test_score_params_fails_without_subtasks(tmp_path: Path, run_cli: RunCli) -> None:
    contest = make_contest(
        tmp_path,
        TaskSpec(
            codename="sum",
            static=True,
            testplan=ABSENT,
            statement="= Statement\n",
        ),
    )
    result = run_cli("score-params", cwd=contest / "sum")
    assert result.exit_code == 2, result.output
    assert "the task has no subtasks." in result.output


def test_outside_contest_is_an_error(tmp_path: Path, run_cli: RunCli) -> None:
    result = run_cli("run-testplan", cwd=tmp_path)
    assert result.exit_code == 1
    assert "ocimatic was not called inside a contest." in result.output


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
def test_subtask_must_be_positive(
    tmp_path: Path,
    run_cli: RunCli,
    args: list[str],
    value: str,
) -> None:
    result = run_cli(*args, value, cwd=tmp_path)
    assert result.exit_code == 2
    assert "x>=1" in result.output


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
def test_subtask_above_number_of_subtasks_is_an_error(
    tmp_path: Path,
    run_cli: RunCli,
    args: list[str],
) -> None:
    # The task has a single subtask.
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    result = run_cli(*args, "2", cwd=contest / "sum")
    assert result.exit_code == 1, result.output
    assert "Subtask 2 doesn't exist: `sum` has 1 subtask." in result.output


@pytest.mark.parametrize(
    "command",
    [
        "run-testplan",
        "validate-input",
        "validate-output",
    ],
)
def test_subtask_with_several_tasks_is_an_error(
    tmp_path: Path,
    run_cli: RunCli,
    command: str,
) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="a"), TaskSpec(codename="b"))
    result = run_cli(command, "--subtask", "1", cwd=contest)
    assert result.exit_code == 1, result.output
    assert (
        "A subtask can only be specified when there's a single target task."
        in result.output
    )
