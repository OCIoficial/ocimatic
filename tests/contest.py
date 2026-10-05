"""Build small contests on disk for tests.

Contests use Python programs only, so tests don't need compilers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from .text import block

# A correct solution for the default task: print the sum of two numbers. Python solutions are
# excluded from runtime stats by default, which `check-dataset` needs, so we include it explicitly.
SUM_PY = block("""
    # @ocimatic::include-in-stats true
    print(sum(map(int, input().split())))
""")


# `#subtask` is normally defined in `oci.typ`; define it here so the statement compiles alone.
# It declares a single subtask, matching `STANDARD_TESTPLAN`.
STANDARD_STATEMENT = block("""
    #let subtask(points) = [Subtask (#points points)]
    = Statement
    #subtask(100)
""")

# A single subtask whose input `SUM_PY` solves.
STANDARD_TESTPLAN = block("""
    [Subtask 1]
      small ; echo 1 2
""")


class Absent(Enum):
    ABSENT = auto()


class Derived(Enum):
    DERIVED = auto()


ABSENT = Absent.ABSENT
"""Don't write this part of the task."""

DERIVED = Derived.DERIVED
"""Build this part from the other fields of the spec."""


@dataclass(frozen=True, kw_only=True)
class TaskSpec:
    """Description of a task.

    `task.toml`, the statement and the testplan have standard content by default and can be left
    out with `ABSENT`. For the directories, only the given files are written: an empty dict creates
    no directory. File contents are written exactly as given.
    """

    codename: str

    static: bool = False
    """Whether the task has a static dataset. Only used when `task_toml` is `DERIVED`."""

    task_toml: str | Derived | Absent = DERIVED
    """Content of `task.toml`. `DERIVED` builds it from `codename` and `static`."""

    statement: str | Absent = STANDARD_STATEMENT
    """Content of `statement/statement.typ`."""

    testplan: str | Absent = STANDARD_TESTPLAN
    """Content of `testplan/testplan.txt`."""

    managers: dict[str, str] = field(default_factory=dict[str, str])
    """Files inside `managers/`."""

    correct: dict[str, str] = field(default_factory=lambda: {"sum.py": SUM_PY})
    """Files inside `solutions/correct/`."""

    partial: dict[str, str] = field(default_factory=dict[str, str])
    """Files inside `solutions/partial/`."""

    dataset: dict[str, str] = field(default_factory=dict[str, str])
    """Files inside `dataset/`, e.g. `{"st1/a.in": "1 2\\n"}`."""

    files: dict[str, str] = field(default_factory=dict[str, str])
    """Any other files, relative to the task directory."""


def make_contest(root: Path, *tasks: TaskSpec, phase: str = "Test") -> Path:
    """Write a minimal contest in `root / "contest"` and return its directory."""
    contest = root / "contest"
    _write(
        contest / "contest.toml",
        block(f"""
            [contest]
            phase = "{phase}"
            typesetting = "typst"
        """),
    )
    _write(contest / "titlepage.typ", "= Titlepage\n")
    _write(contest / "general.typ", "= General\n")
    for task in tasks:
        _make_task(contest / task.codename, task)
    return contest


def _make_task(directory: Path, task: TaskSpec) -> None:
    directory.mkdir(parents=True)
    match task.task_toml:
        case Derived.DERIVED:
            static = "true" if task.static else "false"
            _write(
                directory / "task.toml",
                block(f"""
                    [task]
                    codename = "{task.codename}"

                    [dataset]
                    static = {static}
                """),
            )
        case Absent.ABSENT:
            pass
        case content:
            _write(directory / "task.toml", content)
    _write_file(directory / "statement" / "statement.typ", task.statement)
    _write_file(directory / "testplan" / "testplan.txt", task.testplan)
    _write_dir(directory / "managers", task.managers)
    _write_dir(directory / "solutions" / "correct", task.correct)
    _write_dir(directory / "solutions" / "partial", task.partial)
    _write_dir(directory / "dataset", task.dataset)
    _write_dir(directory, task.files)


def _write_file(path: Path, content: str | Absent) -> None:
    if content is not ABSENT:
        _write(path, content)


def _write_dir(directory: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        _write(directory / name, content)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # `newline=""` writes `\n` as is; otherwise Windows would turn it into `\r\n`.
    path.write_text(content, encoding="utf-8", newline="")
