from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ocimatic import ui
from ocimatic.checkers import Checker
from ocimatic.dataset import Dataset
from ocimatic.source_code import CppSource, JavaSource, SourceCode


class Solution:
    """Abstract class to represent a solution
    """
    def __init__(self, source: SourceCode):
        self._source = source

    @staticmethod
    def load_solutions_in_dir(codename: str, dir: Path, managers_dir: Path) -> List['Solution']:
        """Search for solutions in a directory."""
        assert dir.is_dir()
        return [
            solution for file_path in dir.iterdir()
            for solution in [Solution.load(codename, file_path, managers_dir)] if solution
        ]

    @staticmethod
    def load(codename: str, source_path: Path, managers_dir: Path) -> Optional['Solution']:
        if not source_path.is_file():
            return None

        source: SourceCode
        if source_path.suffix == '.cpp':
            grader = Path(managers_dir, 'grader.cpp')
            source = CppSource(source_path,
                               extra_sources=[grader] if grader.exists() else [],
                               include=managers_dir)
        elif source_path.suffix == '.java':
            source = JavaSource(codename, source_path)
        else:
            return None

        return Solution(source)

    @ui.solution_group()
    def run(self,
            dataset: Dataset,
            checker: Checker,
            check: bool = False,
            sample: bool = False) -> Iterable[ui.WorkResult]:
        """Run this solution for all test cases in the given dataset."""
        runnable = self._source.build()
        success = runnable is not None
        yield ui.WorkResult(success=success, short_msg="OK" if success else "Failed")
        if runnable:
            dataset.run(runnable, checker, sample=sample, check=check)

    @ui.solution_group()
    def gen_expected(self, dataset: Dataset, sample: bool = False) -> Iterable[ui.WorkResult]:
        """Generate expected output files for all test cases in the given dataset
        running this solution."""
        runnable = self._source.build()
        success = runnable is not None
        yield ui.WorkResult(success=success, short_msg="OK" if success else "Failed")
        if runnable is not None:
            dataset.gen_expected(runnable, sample=sample)

    @ui.work('Build')
    def build(self) -> ui.WorkResult:
        """Build solution."""
        success = self._source.build(True) is not None
        msg = 'OK' if success else 'FAILED'
        return ui.WorkResult(success=success, short_msg=msg)

    @property
    def name(self) -> str:
        return self._source.name

    def __str__(self) -> str:
        return str(self.name)

    @property
    def source(self) -> SourceCode:
        return self._source
