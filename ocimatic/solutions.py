from abc import ABC, abstractmethod
from ocimatic.filesystem import Directory, FilePath
from typing import Iterable, List, Optional, Tuple
from ocimatic import ui
from ocimatic.compilers import CppCompiler, JavaCompiler
from ocimatic.runnable import Runnable
from ocimatic.dataset import Dataset
from ocimatic.checkers import Checker


class Solution(ABC):
    """Abstract class to represent a solution
    """
    def __init__(self, source: FilePath):
        self._source = source

    @staticmethod
    def get_solutions(codename: str, solutions_dir: Directory,
                      managers_dir: Directory) -> List['Solution']:
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
            solution for file_path in solutions_dir.lsfile()
            for solution in [Solution.get_solution(codename, file_path, managers_dir)] if solution
        ]

    @staticmethod
    def get_solution(codename: str, file_path: FilePath,
                     managers_dir: Directory) -> Optional['Solution']:
        if file_path.ext == CppSolution.ext:
            return CppSolution(file_path, managers_dir)
        if file_path.ext == JavaSolution.ext:
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

    def get_and_build(self) -> Tuple[Optional[Runnable], str]:
        """
        Returns:
            Optional[Runnable]: Runnable file of this solution or None if it fails
          to build"""
        if self.build_time() < self._source.mtime():
            with ui.capture_io(None), ui.capture_works() as works:
                self.build()
                (st, msg) = works[0]
            if not st:
                return (None, msg)
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

    def __init__(self, source: FilePath, managers: Directory):
        """
        Args:
            source (FilePath): Source code.
            managers (Directory): Directory where managers reside.
        """
        assert source.ext == self.ext
        super().__init__(source)

        self._source = source
        self._compiler = CppCompiler(['-I"%s"' % managers])
        self._grader = managers.find_file('grader.cpp')
        self._bin_path = self._source.chext('.bin')

    def get_runnable(self) -> Runnable:
        return Runnable(self._bin_path)

    def build_time(self) -> float:
        return self._bin_path.mtime()

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

    def __init__(self, codename: str, source: FilePath, managers: Directory):
        """
        Args:
            source (FilePath): Source code.
            managers (Directory): Directory where managers reside.
        """
        # TODO: implement managers for java
        del managers
        super().__init__(source)
        assert source.ext == self.ext
        self._source = source
        self._compiler = JavaCompiler()
        # self._grader = managers.find_file('grader.cpp')
        self._classname = codename
        self._classpath = self._source.directory().path()
        self._bytecode = self._source.chext('.class')

    def get_runnable(self) -> Runnable:
        return Runnable('java', ['-cp', str(self._classpath), '-Xss1g', str(self._classname)])

    def build_time(self) -> float:
        return self._bytecode.mtime()

    def _build(self) -> bool:
        """Compile solution with the JavaCompiler.
        @TODO (NL: 26/09/2016) Compile solutions with a grader if present.
        """
        sources = [self._source]
        # if self._grader:
        #     sources.append(self._grader)
        return self._compiler(sources)
