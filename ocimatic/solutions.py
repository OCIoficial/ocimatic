from ocimatic import ui
from ocimatic.compilers import CppCompiler, JavaCompiler
from ocimatic.runnable import Runnable


class Solution:
    """Abstract class to represent a solution
    """
    def __init__(self, source):
        self._source = source

    @staticmethod
    def get_solutions(codename, solutions_dir, managers_dir):
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
    def get_solution(codename, file_path, managers_dir):
        if file_path.ext == CppSolution.ext:
            return CppSolution(file_path, managers_dir)
        if file_path.ext == JavaSolution.ext:
            return JavaSolution(codename, file_path, managers_dir)
        return None

    @ui.solution_group()
    def run(self, dataset, checker, check=False, sample=False):
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
        yield (runnable is not None, msg)
        if runnable:
            dataset.run(runnable, checker, sample=sample, check=check)

    @ui.solution_group()
    def gen_expected(self, dataset, sample=False):
        """Generate expected output files for all test cases in the given dataset
        running this solution.
        Args:
            dataset (Dataset)
            sample (bool): If true expected output file for are generated for
                sample test data from statement.
        """
        runnable, msg = self.get_and_build()
        yield (runnable is not None, msg)
        if runnable:
            dataset.gen_expected(runnable, sample=sample)

    def _build(self):
        raise NotImplementedError("Class %s doesn't implement get_runnable()" %
                                  (self.__class__.__name__))

    @ui.work('Build', verbosity=False)
    def build(self):
        """Build solution.
        Returns:
            (bool, str): A tuple containing status and result message.
        """
        st = self._build()
        msg = 'OK' if st else 'FAILED'
        return (st, msg)

    def get_and_build(self):
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

    def get_runnable(self):
        raise NotImplementedError("Class %s doesn't implement get_runnable()" %
                                  (self.__class__.__name__))

    def build_time(self):
        raise NotImplementedError("Class %s doesn't implement build_time()" %
                                  (self.__class__.__name__))

    @property
    def name(self):
        return self._source.name

    def __str__(self):
        return str(self._source)


class CppSolution(Solution):
    """Solution written in C++. This solutions is compiled with
    a grader if one is present in the managers directory.
    """
    ext = '.cpp'

    def __init__(self, source, managers):
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

    def get_runnable(self):
        return Runnable(self._bin_path)

    def build_time(self):
        return self._bin_path.mtime()

    def _build(self):
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

    def __init__(self, codename, source, managers):
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

    def get_runnable(self):
        return Runnable('java', ['-cp', str(self._classpath), str(self._classname)])

    def build_time(self):
        return self._bytecode.mtime()

    def _build(self):
        """Compile solution with the JavaCompiler.
        @TODO (NL: 26/09/2016) Compile solutions with a grader if present.
        """
        sources = [self._source]
        # if self._grader:
        #     sources.append(self._grader)
        return self._compiler(sources)
