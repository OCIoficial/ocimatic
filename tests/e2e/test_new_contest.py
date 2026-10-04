"""End-to-end tests that drive the installed `ocimatic` command."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

DATA_DIR = Path(__file__).parent / "data"


def _ocimatic_executable() -> str:
    # Prefer the script installed next to the running interpreter so we test the same
    # environment pytest runs in, and fall back to whatever is on PATH.
    exe = shutil.which("ocimatic", path=str(Path(sys.executable).parent))
    exe = exe or shutil.which("ocimatic")
    if exe is None:
        pytest.fail("`ocimatic` executable not found; install the package first")
    return exe


def _run(cwd: Path, *args: str) -> None:
    cmd = [_ocimatic_executable(), *args]
    complete = subprocess.run(
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if complete.returncode != 0:
        pytest.fail(
            f"`ocimatic {' '.join(args)}` exited with code {complete.returncode}\n"
            f"--- stdout ---\n{complete.stdout}\n"
            f"--- stderr ---\n{complete.stderr}",
            pytrace=False,
        )


def test_new_contest_passes_check_dataset(tmp_path: Path) -> None:
    _run(tmp_path, "init", "contest", "--phase", "Test", "--typesetting", "typst")
    contest = tmp_path / "contest"

    _run(contest, "new-task", "task")
    shutil.copy(DATA_DIR / "task.java", contest / "task" / "solutions" / "correct")

    _run(contest, "run-testplan")
    _run(contest, "gen-expected")
    _run(contest, "check-dataset")
