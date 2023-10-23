from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ocimatic.runnable import RunSuccess
from ocimatic.source_code import BuildError, CppSource, RustSource, SourceCode


@dataclass
class CheckerSuccess:
    outcome: float
    msg: Optional[str] = None


@dataclass
class CheckerError:
    msg: str


CheckerResult = CheckerSuccess | CheckerError


class Checker(ABC):
    """Check solutions
    """

    @abstractmethod
    def run(self, in_path: Path, expected_path: Path, out_path: Path) -> CheckerResult:
        raise NotImplementedError("Class %s doesn't implement run()" % (self.__class__.__name__))

    @staticmethod
    def find_in_directory(dir: Path) -> 'Checker':
        for f in dir.iterdir():
            if f.name == 'checker.cpp':
                return CustomChecker(CppSource(f, include=dir, out=Path(dir, 'checker')))
            elif f.name == 'checker.rs':
                return CustomChecker(RustSource(f, out=Path(dir, 'checker')))
        return DiffChecker()


class DiffChecker(Checker):
    """White diff checker
    """

    def run(self, in_path: Path, expected_path: Path, out_path: Path) -> CheckerResult:
        """Performs a white diff between expected output and output files.
        Parameters correspond to convention for checker in cms.
        Args:
            in_path (FilePath)
            expected_path (FilePath)
            out_path (FilePath)
        """
        assert in_path.exists()
        assert expected_path.exists()
        assert out_path.exists()

        expected = open(expected_path).readlines()
        out = open(out_path).readlines()
        if len(expected) != len(out):
            return CheckerSuccess(outcome=0.0)

        for (a, b) in zip(expected, out):
            if a != b:
                return CheckerSuccess(outcome=0.0)
        return CheckerSuccess(outcome=1.0)


class CustomChecker(Checker):

    def __init__(self, code: SourceCode):
        """
        Args:
            source (FilePath)
        """
        self._code = code

    def run(self, in_path: Path, expected_path: Path, out_path: Path) -> CheckerResult:
        """Run checker to evaluate outcome. Parameters correspond to convention
        for checker in cms.
        Args:
            in_path (FilePath)
            expected_path (FilePath)
            out_path (FilePath)
        """

        assert in_path.exists()
        assert expected_path.exists()
        assert out_path.exists()
        build_result = self._code.build()
        if isinstance(build_result, BuildError):
            return CheckerError(msg="Failed to build checker")
        result = build_result.run(args=[str(in_path), str(expected_path), str(out_path)])
        if isinstance(result, RunSuccess):
            try:
                stderr = result.stderr.strip()
                msg = stderr if stderr != "" else None
                return CheckerSuccess(outcome=float(result.stdout), msg=msg)
            except ValueError:
                return CheckerError(msg='output must be a valid float')
        else:
            return CheckerError(msg=result.msg)
