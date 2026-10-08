from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from ocimatic.runnable import RunSuccess
from ocimatic.source_code import BuildError, CppSource, RustSource, SourceCode


@dataclass
class CheckerSuccess:
    outcome: float
    msg: str | None = None


@dataclass
class CheckerError:
    msg: str


type CheckerResult = CheckerSuccess | CheckerError


class Checker(ABC):
    """Abstract class for a checker."""

    @abstractmethod
    def run(
        self,
        *,
        in_path: Path,
        expected_path: Path,
        out_path: Path,
    ) -> CheckerResult:
        raise NotImplementedError(
            f"Class {self.__class__.__name__} doesn't implement run()",
        )

    @staticmethod
    def find_in_directory(dir: Path) -> Checker:
        """Find a custom checker in `dir`, falling back to `DiffChecker` (also if `dir` is missing)."""
        if not dir.is_dir():
            return DiffChecker()
        for f in dir.iterdir():
            if f.name == "checker.cpp":
                return CustomChecker(
                    CppSource(f, include=dir, out=Path(dir, "checker")),
                )
            elif f.name == "checker.rs":
                return CustomChecker(RustSource(f, out=Path(dir, "checker")))
        return DiffChecker()


class DiffChecker(Checker):
    """White diff checker, matching CMS's default comparator (`cms/grading/steps/whitediff.py`)."""

    def run(
        self,
        *,
        in_path: Path,
        expected_path: Path,
        out_path: Path,
    ) -> CheckerResult:
        """Perform a white diff between expected output and output files.

        Parameters correspond to convention for checker in cms.
        """
        assert in_path.exists()
        assert expected_path.exists()
        assert out_path.exists()

        with out_path.open("rb") as out_file, expected_path.open("rb") as expected_file:
            return _white_diff(out_file, expected_file)


# The whitespace characters of CMS's white diff.
_WHITES = b" \t\n\x0b\x0c\r"

# Lines longer than this are shortened in mismatch messages.
_LENGTH_LIMIT = 100


def _white_diff(out_file: BinaryIO, expected_file: BinaryIO) -> CheckerSuccess:
    r"""Compare files line by line, ignoring differences in the number or kind of whitespace.

    Lines end only at `\n`. Trailing lines that contain only whitespace are ignored.
    """
    line = 0
    while True:
        out = out_file.readline()
        expected = expected_file.readline()
        line += 1

        if not out and not expected:
            return CheckerSuccess(outcome=1.0)

        if not out or not expected:
            # One file ended; the rest of the other one may only contain whitespace.
            if out.strip(_WHITES):
                return CheckerSuccess(outcome=0.0, msg="Contestant output too long")
            if expected.strip(_WHITES):
                return CheckerSuccess(outcome=0.0, msg="Contestant output too short")
            continue

        out = _canonicalize(out)
        expected = _canonicalize(expected)
        if out != expected:
            return CheckerSuccess(
                outcome=0.0,
                msg=f"Expected `{_shorten(expected)}`, found `{_shorten(out)}` on line {line}",
            )


def _canonicalize(line: bytes) -> bytes:
    """Strip whitespace at both ends and collapse each run of whitespace into a single space."""
    for char in _WHITES[1:]:
        line = line.replace(bytes([char]), b" ")
    return b" ".join(token for token in line.split(b" ") if token)


def _shorten(line: bytes) -> str:
    if len(line) > _LENGTH_LIMIT:
        line = line[:_LENGTH_LIMIT] + b"..."
    return line.decode("utf-8", errors="backslashreplace")


class CustomChecker(Checker):
    def __init__(self, code: SourceCode) -> None:
        self._code = code

    def run(
        self,
        *,
        in_path: Path,
        expected_path: Path,
        out_path: Path,
    ) -> CheckerResult:
        """Run custom checker to evaluate outcome.

        Parameters correspond to convention for checker in cms.
        """
        assert in_path.exists()
        assert expected_path.exists()
        assert out_path.exists()
        build_result = self._code.build()
        if isinstance(build_result, BuildError):
            return CheckerError(msg="Failed to build checker")
        args = [str(in_path), str(expected_path), str(out_path)]
        result = build_result.run(args=args)
        if isinstance(result, RunSuccess):
            try:
                stderr = result.stderr.strip()
                msg = stderr if stderr != "" else None
                return CheckerSuccess(outcome=float(result.stdout), msg=msg)
            except ValueError:
                return CheckerError(msg="output must be a valid float")
        else:
            return CheckerError(msg=result.msg)
