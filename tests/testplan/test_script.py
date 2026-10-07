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


@pytest.mark.xfail(strict=True, reason="attic/unresolved-problems.md #6")
def test_generator_writing_a_lot_to_stderr_does_not_deadlock(tmp_path: Path) -> None:
    contest = make_contest(
        tmp_path,
        TaskSpec(
            codename="sum",
            testplan=block("""
                [Subtask 1]
                  rand ; gen.py
            """),
            files={"testplan/gen.py": LOUD_GENERATOR},
        ),
    )

    # Run in a subprocess so the deadlock can be timed out: killing ocimatic closes the stderr pipe,
    # which also ends the blocked generator.
    try:
        complete = subprocess.run(
            [sys.executable, "-c", "import ocimatic; ocimatic.main()", "run-testplan"],
            cwd=contest / "sum",
            # Don't read the user's `~/.ocimatic.toml`.
            env={
                **os.environ,
                "HOME": str(tmp_path / "home"),
                "USERPROFILE": str(tmp_path / "home"),
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"`ocimatic run-testplan` didn't finish within {TIMEOUT} seconds")

    assert complete.returncode == 0, complete.stdout
    assert read_tree(contest / "sum" / "dataset") == {"st1": {"rand-1.in": "1 2\n"}}
