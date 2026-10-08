"""Build small contests on disk for tests.

Contests use Python programs only, so tests don't need compilers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from .text import block
from .tree import Tree, write_file, write_tree

# A correct solution for the default task: print the sum of two numbers. Python solutions are
# excluded from runtime stats by default, which `check-dataset` needs, so we include it explicitly.
SUM_PY = block("""
    # @ocimatic::include-in-stats true
    print(sum(map(int, input().split())))
""")

# The default solutions of a task, relative to `solutions/`.
STANDARD_SOLUTIONS: Tree = {"correct/sum.py": SUM_PY}


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
    out with `ABSENT`. The directories are trees written with `write_tree`: only the given entries
    are written, so an empty tree creates no directory. File contents are written exactly as given.
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

    managers: Tree = field(default_factory=dict[str, str])
    """Files inside `managers/`."""

    solutions: Tree = field(default_factory=lambda: dict(STANDARD_SOLUTIONS))
    """Files inside `solutions/`, e.g. `{"correct/sum.py": SUM_PY, "partial/slow.py": ...}`."""

    dataset: Tree = field(default_factory=dict[str, str])
    """Files inside `dataset/`, e.g. `{"st1/a.in": "1 2\\n"}`."""

    files: Tree = field(default_factory=dict[str, str])
    """Any other files, relative to the task directory."""


def make_contest(root: Path, *tasks: TaskSpec, phase: str = "Test") -> Path:
    """Write a minimal contest in `root / "contest"` and return its directory."""
    contest = root / "contest"
    write_file(
        contest / "contest.toml",
        block(f"""
            [contest]
            phase = "{phase}"
            typesetting = "typst"
        """),
    )
    write_file(contest / "titlepage.typ", "= Titlepage\n")
    write_file(contest / "general.typ", "= General\n")
    for task in tasks:
        _make_task(contest / task.codename, task)
    return contest


def _make_task(directory: Path, task: TaskSpec) -> None:
    directory.mkdir(parents=True)
    match task.task_toml:
        case Derived.DERIVED:
            static = "true" if task.static else "false"
            write_file(
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
            write_file(directory / "task.toml", content)
    _write_optional(directory / "statement" / "statement.typ", task.statement)
    _write_optional(directory / "testplan" / "testplan.txt", task.testplan)
    write_tree(directory / "managers", task.managers)
    write_tree(directory / "solutions", task.solutions)
    write_tree(directory / "dataset", task.dataset)
    write_tree(directory, task.files)


def _write_optional(path: Path, content: str | Absent) -> None:
    if content is not ABSENT:
        write_file(path, content)
