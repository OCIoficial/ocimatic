from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ocimatic import ui
from ocimatic.checkers import Checker
from ocimatic.compilers import CppCompiler, JavaCompiler
from ocimatic.dataset import Dataset
from ocimatic.runnable import Runnable


class Solution(ABC):
    """Abstract class to represent a solution
    """
    def __init__(self, source: Path):
        self._source = source

    @staticmethod
    def get_solutions(codename: str, solutions_dir: Path, managers_dir: Path) -> List['Solution']:
        """Search for solutions in a directory.

        Args:
            solutions_dir (Directory): Directory to look for solutions.
            managers_dir (Directory): Directory where managers reside.
                This is used to provide necessary files for compilation,
                for example, when solutions are compiled with a grader.

        Returns:
            List[Solution]: List of solutions.
        """
        return [
            solution for file_path in solutions_dir.iterdir()
            for solution in [Solution.get_solution(codename, file_path, managers_dir)] if solution
        ]

    @staticmethod
    def get_solution(codename: str, file_path: Path, managers_dir: Path) -> Optional['Solution']:
        if file_path.suffix == CppSolution.ext:
            return CppSolution(file_path, managers_dir)
        if file_path.suffix == JavaSolution.ext:
            return JavaSolution(codename, file_path, managers_dir)
        return None

    @ui.solution_group()
    def run(self,
            dataset: Dataset,
            checker: Checker,
            check: bool = False,
            sample: bool = False) -> Iterable[ui.WorkResult]:
        """Run this solution for all test cases in the given dataset.
        Args:
            dataset (Dataset)
            checker (Checker): Checker to compute outcome.
            check  (bool): If true only report if expected output
                corresponds to solution output.
            sample (bool): If true run solution with sample test data from
                statement.
        """
        runnable, msg = self.get_and_build()
        yield ui.WorkResult(success=runnable is not None, short_msg=msg)
        if runnable:
            dataset.run(runnable, checker, sample=sample, check=check)

    @ui.solution_group()
    def gen_expected(self, dataset: Dataset, sample: bool = False) -> Iterable[ui.WorkResult]:
        """Generate expected output files for all test cases in the given dataset
        running this solution.
        Args:
            dataset (Dataset)
            sample (bool): If true expected output file for are generated for
                sample test data from statement.
        """
        runnable, msg = self.get_and_build()
        yield ui.WorkResult(success=runnable is not None, short_msg=msg)
        if runnable:
            dataset.gen_expected(runnable, sample=sample)

    @ui.work('Build')
    def build(self) -> ui.WorkResult:
        """Build solution.
        Returns:
            (bool, str): A tuple containing status and result message.
        """
        st = self._build()
        msg = 'OK' if st else 'FAILED'
        return ui.WorkResult(success=st, short_msg=msg)

    def get_and_build(self) -> Tuple[Optional[Runnable], Optional[str]]:
        """
        Returns:
            Optional[Runnable]: Runnable file of this solution or None if it fails
          to build"""
        if self.build_time() < self._source.stat().st_mtime:
            with ui.capture_io(None), ui.capture_works() as works:
                self.build()
                result = works[0]
            if not result.success:
                return (None, result.short_msg)
        return (self.get_runnable(), 'OK')

    @abstractmethod
    def get_runnable(self) -> Optional[Runnable]:
        raise NotImplementedError("Class %s doesn't implement get_runnable()" %
                                  (self.__class__.__name__))

    @abstractmethod
    def _build(self) -> bool:
        raise NotImplementedError("Class %s doesn't implement get_runnable()" %
                                  (self.__class__.__name__))

    @abstractmethod
    def build_time(self) -> float:
        raise NotImplementedError("Class %s doesn't implement build_time()" %
                                  (self.__class__.__name__))

    @property
    def name(self) -> str:
        return self._source.name

    def __str__(self) -> str:
        return str(self._source)


class CppSolution(Solution):
    """Solution written in C++. This solutions is compiled with
    a grader if one is present in the managers directory.
    """
    ext = '.cpp'

    def __init__(self, source: Path, managers: Path):
        """
        Args:
            source (FilePath): Source code.
            managers (Directory): Directory where managers reside.
        """
        assert source.suffix == self.ext
        super().__init__(source)

        self._source = source
        self._compiler = CppCompiler(['-I"%s"' % managers])
        self._grader = next(managers.glob('grader.cpp'), None)
        self._bin_path = self._source.with_suffix('.bin')

    def get_runnable(self) -> Runnable:
        return Runnable(self._bin_path)

    def build_time(self) -> float:
        if self._bin_path.exists():
            return self._bin_path.stat().st_mtime
        else:
            return float("-inf")

    def _build(self) -> bool:
        """Compile solution with a CppCompiler. Solutions is compiled with a
        grader if present.
        """
        sources = [self._source]
        if self._grader:
            sources.append(self._grader)
        return self._compiler(sources, self._bin_path)


class JavaSolution(Solution):
    """Solution written in C++. This solutions is compiled with
    a grader if one is present in the managers directory.
    """
    ext = '.java'

    def __init__(self, codename: str, source: Path, managers: Path):
        """
        Args:
            source (FilePath): Source code.
            managers (Directory): Directory where managers reside.
        """
        # TODO: implement managers for java
        del managers
        super().__init__(source)
        assert source.suffix == self.ext
        self._source = source
        self._compiler = JavaCompiler()
        # self._grader = managers.find_file('grader.cpp')
        self._classname = codename
        self._classpath = self._source.parent
        self._bytecode = self._source.with_suffix('.class')

    def get_runnable(self) -> Runnable:
        return Runnable('java', ['-cp', str(self._classpath), '-Xss1g', str(self._classname)])

    def build_time(self) -> float:
        return self._bytecode.stat().st_mtime

    def _build(self) -> bool:
        """Compile solution with the JavaCompiler.
        @TODO (NL: 26/09/2016) Compile solutions with a grader if present.
        """
        sources = [self._source]
        # if self._grader:
        #     sources.append(self._grader)
        return self._compiler(sources)
