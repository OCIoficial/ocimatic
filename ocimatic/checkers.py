from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ocimatic.runnable import RunSuccess
from ocimatic.source_code import BuildError, CppSource, RustSource, SourceCode


@dataclass
class CheckerSuccess:
    outcome: float
    msg: str | None = None


@dataclass
class CheckerError:
    msg: str


CheckerResult = CheckerSuccess | CheckerError


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
        for f in dir.iterdir():
            if f.name == "checker.cpp":
                return CustomChecker(
                    CppSource(f, include=dir, out=Path(dir, "checker")),
                )
            elif f.name == "checker.rs":
                return CustomChecker(RustSource(f, out=Path(dir, "checker")))
        return DiffChecker()


class DiffChecker(Checker):
    """White diff checker."""

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

        with out_path.open() as expected_file, expected_path.open() as output_file:
            expected = expected_file.readlines()
            out = output_file.readlines()

            _filter_trailing_empty_lines(expected)
            _filter_trailing_empty_lines(out)

            if len(out) != len(expected):
                return CheckerSuccess(outcome=0.0)

            # Lines must be equal up to whitespaces
            for i, line in enumerate(expected):
                if line.split() != out[i].split():
                    return CheckerSuccess(outcome=0.0)
            return CheckerSuccess(outcome=1.0)


def _filter_trailing_empty_lines(lines: list[str]) -> None:
    for i in reversed(range(len(lines))):
        if not lines[i].strip():
            lines.pop(i)
        else:
            break


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
