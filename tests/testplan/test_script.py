"""Tests for running generator scripts from a testplan."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ..contest import TaskSpec, make_contest
from ..text import block
from ..tree import read_tree

# Long enough to run a small testplan; a deadlocked run never finishes.
TIMEOUT = 10

# More than the pipe buffer (64 KB on Linux and macOS), so the write blocks until someone reads it.
LOUD_GENERATOR = block("""
    import sys
    sys.stderr.write("x" * 1_000_000)
    print("1 2")
""")

# Writes three tests separated by `FS` (chr 28). An empty test between two `FS` is skipped.
SPLITTING_GENERATOR = block("""
    import sys
    FS = chr(28)
    sys.stdout.write("a\\n" + FS + FS + "b\\n" + FS + "c\\n")
""")

# Writes a test and then fails.
CRASHING_GENERATOR = block("""
    import sys
    print("1 2")
    sys.exit(1)
""")


def test_generator_writing_a_lot_to_stderr_does_not_deadlock(tmp_path: Path) -> None:
    contest = _make_contest_with_generator(tmp_path, LOUD_GENERATOR)

    complete = _run_testplan(contest / "sum", tmp_path)

    assert complete.returncode == 0, complete.stdout
    assert read_tree(contest / "sum" / "dataset") == {
        "st1": {"rand-1.in": _native("1 2\n")},
    }


def test_generator_output_is_split_on_fs(tmp_path: Path) -> None:
    contest = _make_contest_with_generator(tmp_path, SPLITTING_GENERATOR)

    complete = _run_testplan(contest / "sum", tmp_path)

    assert complete.returncode == 0, complete.stdout
    assert read_tree(contest / "sum" / "dataset") == {
        "st1": {
            "rand-1.in": _native("a\n"),
            "rand-2.in": _native("b\n"),
            "rand-3.in": _native("c\n"),
        },
    }


def test_failed_generator_writes_no_tests(tmp_path: Path) -> None:
    contest = _make_contest_with_generator(tmp_path, CRASHING_GENERATOR)

    complete = _run_testplan(contest / "sum", tmp_path)

    assert complete.returncode == 2, complete.stdout
    assert read_tree(contest / "sum" / "dataset") == {"st1": {}}


def _native(text: str) -> str:
    # Generated tests use the platform's line endings, which testlib validators built on Windows
    # require.
    return text.replace("\n", os.linesep)


def _make_contest_with_generator(root: Path, generator: str) -> Path:
    return make_contest(
        root,
        TaskSpec(
            codename="sum",
            testplan=block("""
                [Subtask 1]
                  rand ; gen.py
            """),
            files={"testplan/gen.py": generator},
        ),
    )


def _run_testplan(task: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    # Run in a subprocess so a deadlock can be timed out: killing ocimatic closes the stderr pipe,
    # which also ends a blocked generator.
    try:
        return subprocess.run(
            [sys.executable, "-c", "import ocimatic; ocimatic.main()", "run-testplan"],
            cwd=task,
            # Don't read the user's `~/.ocimatic.toml`.
            env={
                **os.environ,
                "HOME": str(tmp_path / "home"),
                "USERPROFILE": str(tmp_path / "home"),
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"`ocimatic run-testplan` didn't finish within {TIMEOUT} seconds")
