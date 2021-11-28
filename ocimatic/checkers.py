import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple
from ocimatic.runnable import RunSuccess

from ocimatic.source_code import BuildError, CppSource


class CheckerResult(NamedTuple):
    success: bool
    outcome: float
    msg: str


class Checker(ABC):
    """Check solutions
    """
    @abstractmethod
    def run(self, in_path: Path, expected_path: Path, out_path: Path) -> CheckerResult:
        raise NotImplementedError("Class %s doesn't implement run()" % (self.__class__.__name__))


class DiffChecker(Checker):
    """White diff checker
    """
    def run(self, in_path: Path, expected_path: Path, out_path: Path) -> CheckerResult:
        """Performs a white diff between expected output and output files
        Parameters correspond to convention for checker in cms.
        Args:
            in_path (FilePath)
            expected_path (FilePath)
            out_path (FilePath)
        """
        assert shutil.which('diff')
        assert in_path.exists()
        assert expected_path.exists()
        assert out_path.exists()
        complete = subprocess.run(
            ['diff', str(expected_path), str(out_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False)
        # @TODO(NL 20/10/1990) Check if returncode is nonzero because there
        # is a difference between files or because there was an error executing
        # diff.
        st = complete.returncode == 0
        outcome = 1.0 if st else 0.0
        return CheckerResult(success=True, outcome=outcome, msg='')


class CppChecker(Checker):
    def __init__(self, source: Path):
        """
        Args:
            source (FilePath)
        """
        self._source = CppSource(source, include=source.parent, out=Path(source.parent, 'checker'))

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
        build_result = self._source.build()
        if isinstance(build_result, BuildError):
            return CheckerResult(success=False, outcome=0.0, msg="Failed to build checker")
        result = build_result.run(args=[str(in_path), str(expected_path), str(out_path)])
        if isinstance(result, RunSuccess):
            success = True
            msg = ''
            try:
                outcome = float(result.stdout)
            except ValueError:
                outcome = 0.0
                msg = 'Output must be a valid float'
                success = False
            return CheckerResult(success=success, outcome=outcome, msg=msg)
        else:
            msg = result.msg
            outcome = 0.0
            return CheckerResult(success=True, outcome=outcome, msg=msg)
