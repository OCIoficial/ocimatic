"""Operations on contests and tasks report failures through the `Status` they return."""

from __future__ import annotations

from pathlib import Path

import pytest

from ocimatic.core import Contest
from ocimatic.result import Status

from ..conftest import UseEnv
from ..contest import ABSENT, TaskSpec, make_contest
from ..text import block

# A task with a static dataset that `check-dataset` accepts.
STATIC_DATASET = {"st1/a.in": "1 2\n", "st1/a.sol": "3\n"}


def test_missing_solution_directories_load_as_empty(
    tmp_path: Path,
    use_env: UseEnv,
) -> None:
    # Empty dicts mean `solutions/correct/` and `solutions/partial/` aren't created.
    contest = make_contest(tmp_path, TaskSpec(codename="sum", correct={}, partial={}))
    with use_env(cwd=contest, contest_root=contest):
        [task] = Contest.load().tasks
        assert task.run_testplan(stn=None) == Status.success
        # Without correct solutions there's nothing to generate expected output with.
        assert task.gen_expected() == Status.fail


def test_compress_empty_dataset_fails(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    with use_env(cwd=contest, contest_root=contest):
        [task] = Contest.load().tasks
        assert task.compress_dataset(random_sort=False) == Status.fail


def test_build_statement_succeeds(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum"))
    with use_env(cwd=contest, contest_root=contest):
        [task] = Contest.load().tasks
        assert task.build_statement() == Status.success


def test_build_statement_fails_on_typst_error(tmp_path: Path, use_env: UseEnv) -> None:
    contest = make_contest(tmp_path, TaskSpec(codename="sum", statement="#broken(\n"))
    with use_env(cwd=contest, contest_root=contest):
        [task] = Contest.load().tasks
        assert task.build_statement() == Status.fail


def test_archive_fails_when_a_statement_fails(
    tmp_path: Path,
    use_env: UseEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contest = make_contest(
        tmp_path,
        TaskSpec(
            codename="sum",
            statement="#broken(\n",
            static=True,
            testplan=ABSENT,
            dataset=STATIC_DATASET,
        ),
    )
    # `archive` writes `archive.zip` to the working directory.
    monkeypatch.chdir(tmp_path)
    with use_env(cwd=contest, contest_root=contest):
        assert Contest.load().archive() == Status.fail


def test_normalize_leaves_normalized_files_untouched(
    tmp_path: Path,
    use_env: UseEnv,
) -> None:
    contest = make_contest(
        tmp_path,
        TaskSpec(codename="sum", static=True, testplan=ABSENT, dataset=STATIC_DATASET),
    )
    in_path = contest / "sum" / "dataset" / "st1" / "a.in"
    mtime = in_path.stat().st_mtime_ns
    with use_env(cwd=contest, contest_root=contest):
        [task] = Contest.load().tasks
        assert task.normalize() == Status.success
    assert in_path.stat().st_mtime_ns == mtime


def test_check_dataset_fails_when_correct_solutions_crash(
    tmp_path: Path,
    use_env: UseEnv,
) -> None:
    contest = make_contest(
        tmp_path,
        TaskSpec(
            codename="sum",
            static=True,
            testplan=ABSENT,
            dataset=STATIC_DATASET,
            correct={
                "crash.py": block("""
                    # @ocimatic::include-in-stats true
                    raise SystemExit(1)
                """),
            },
        ),
    )
    with use_env(cwd=contest, contest_root=contest):
        [task] = Contest.load().tasks
        assert task.check_dataset() == Status.fail
