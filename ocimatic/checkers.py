from abc import ABC, abstractmethod
import shutil
import subprocess
from typing import NamedTuple, Tuple

from ocimatic.compilers import CppCompiler
from ocimatic.filesystem import FilePath
from ocimatic.runnable import SIGNALS


class CheckerResult(NamedTuple):
    success: bool
    outcome: float
    msg: str


class Checker(ABC):
    """Check solutions
    """
    @abstractmethod
    def __call__(self, in_path: FilePath, expected_path: FilePath,
                 out_path: FilePath) -> CheckerResult:
        """Check outcome.

        Args:
            in_path (FilePath): Input file.
            expected_path (FilePath): Expected solution file
            out_path (FilePath): Output file.

        Returns:
            float: Float between 0.0 and 1.0 indicating result.
        """
        raise NotImplementedError("Class %s doesn't implement __call__()" %
                                  (self.__class__.__name__))


class DiffChecker(Checker):
    """White diff checker
    """
    def __call__(self, in_path: FilePath, expected_path: FilePath,
                 out_path: FilePath) -> CheckerResult:
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
    def __init__(self, source: FilePath):
        """
        Args:
            source (FilePath)
        """
        self._source = source
        self._compiler = CppCompiler(['-I"%s"' % source.directory()])
        self._binary_path = FilePath(source.directory(), 'checker')

    def __call__(self, in_path: FilePath, expected_path: FilePath,
                 out_path: FilePath) -> CheckerResult:
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
        if self._binary_path.mtime() < self._source.mtime():
            if not self.build():
                return CheckerResult(success=False, outcome=0.0, msg="Failed to build checker")
        complete = subprocess.run(
            [str(self._binary_path),
             str(in_path), str(expected_path),
             str(out_path)],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False)
        ret = complete.returncode
        st = ret == 0
        if st:
            try:
                outcome = float(complete.stdout)
                msg = complete.stderr
            except ValueError:
                outcome = 0.0
                msg = 'Output must be a valid float'
                st = False
        else:
            stderr = complete.stderr.strip('\n')
            outcome = 0.0
            if stderr and len(stderr) < 75:
                msg = stderr
            else:
                if ret < 0:
                    sig = -ret
                    msg = 'Execution killed with signal %d' % sig
                    if sig in SIGNALS:
                        msg += ': %s' % SIGNALS[sig]
                else:
                    msg = 'Execution ended with error (return code %d)' % ret

        return CheckerResult(success=st, outcome=outcome, msg=msg)

    def build(self) -> bool:
        """Build source of the checker
        Returns:
            bool: True if compilation is successful. False otherwise
        """
        return self._compiler(self._source, self._binary_path)
