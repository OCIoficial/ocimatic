# coding=UTF-8
import os
import glob
import subprocess
from functools import total_ordering
from tempfile import mkstemp


class Contest(object):
    """This class represent a contest. A contest is conformed by a list of
    problems and a titlepage. In ocimatic a contest is always associated to
    a directory in the filesystem.
    """
    def __init__(self, directory):
        """
        Args:
            directory (Directory): Directory where the contest reside.
        """
        self._directory = directory

    def problemset(self):
        pass

    def get_problem_by_name(self, name):
        pass


class Problem(object):
    """This class represent a problem. A problem consist of a statement,
    a list of correct and partial solutions and a dataset. A problem is
    always associated to a directory in the filesystem.

    """
    def __init__(self, directory, num):
        """
        Args:
            directory (Directory): Directory where the problem reside.
            num (int): Position of the problem in the problemset.
        """
        self._directory = directory
        self._num = num

        managers_dir = directory.chdir('managers')

        correct_dir = directory.chdir('solutions/correct')
        self._correct = Solution.get_solutions(correct_dir, managers_dir)
        partial_dir = directory.chdir('solutions/partial')
        self._partial = Solution.get_solutions(partial_dir, managers_dir)

        self._checker = DiffChecker()

        self._dataset = DataSet(directory.chdir('dataset'))

    def run_solutions(self, partial=False):
        solutions = self._correct
        if partial:
            solutions.extend(self._partial)

        for sol in solutions:
            self._dataset.run(sol.binary, self._checker)


class Solution(object):
    """Abstract class to represent a solution
    """
    @staticmethod
    def get_solutions(solutions_dir, managers_dir):
        """Search for solutions in a directory.

        Args:
            solutions_dir (Directory): Directory to look for solutions.
            managers_dir (Directory): Directory where managers reside.
                This is used for example when solutions are compiled
                with a grader.

        Returns:
            List[Solution]: List of solutions.
        """
        solutions = []
        for f in solutions_dir.lsfile():
            if f.ext == CppSolution.ext:
                solutions.append(CppSolution(f, managers_dir))
        return solutions

    def build(self):
        NotImplementedError("Class %s doesn't implement build()" % (
            self.__class__.__name__))

    @property
    def binary(self):
        # TODO what happens if building fails?
        if self._bin_path.mtime() < self._source.mtime():
            print('Rebuild')
            self.build()
        return Binary(self._bin_path)


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
        assert(source.ext == self.ext)
        self._source = source
        self._compiler = CppCompiler()
        self._grader = managers.find_file('grader.cpp')
        self._bin_path = self._source.chext('.bin')

    def build(self):
        sources = [self._source]
        if self._grader:
            sources.append(self._grader)
        return self._compiler(sources, self._bin_path)


class CppCompiler(object):
    """Compiles C++ code
    """
    def __init__(self):
        self._cmd_template = 'g++ -std=c++11 -O2 -o %s %s'

    def __call__(self, sources, out):
        """Compiles a list of sources

        Args:
            sources (List[FilePath]): List of sources.
            out (FilePath):
        """
        out = '"%s"' % out.fullpath
        sources = ' '.join('"%s"' % w for w in sources)
        cmd = self._cmd_template % (out, sources)
        return subprocess.call(cmd, shell=True) == 0


class DataSet(object):
    """Test data"""
    def __init__(self, directory):
        """
        Args:
            directory (Directory)
        """
        self._tests = [Test(f) for f in directory.lsfile('*/*.in')]

    # TODO documentation and error handling
    def run(self, binary, checker):
        print(str(binary))
        print('########################################################')
        for test in self._tests:
            print(str(test))
            tmp_path = mkstemp()[1]
            out_path = FilePath(tmp_path)
            if test.expected_path:
                binary.run(test.in_path, out_path)
                outcome = checker(test.in_path,
                                  test.expected_path,
                                  out_path)
                print(outcome)
            else:
                print('No expected output file')
            os.remove(tmp_path)
            print('-----')

    def compress(self):
        pass


class Test(object):
    """A single test file. Expected output file may not exists"""
    def __init__(self, in_path):
        """
        Args:
            in_path (FilePath)
        """
        assert(in_path.exists())
        self._in_path = in_path
        self._expected_path = in_path.chext('.sol')

    def __str__(self):
        return str(self._in_path)

    @property
    def in_path(self):
        """FilePath: Input file path"""
        return self._in_path

    @property
    def expected_path(self):
        """FilePath: Expected output file path."""
        return self._expected_path


class Checker(object):
    """Check solutions
    """
    def __call__(self, in_path, expected_path, out_path):
        """Check output of solution.

        Args:
            in_path (FilePath): Input file.
            expected_path (FilePath): Expected solution file
            out_path (FilePath): Output file.

        Returns:
            float: Float between 0 and 1 indicating result.
        """
        NotImplementedError("Class %s doesn't implement __call__()" % (
            self.__class__.__name__))


class DiffChecker(Checker):
    """White diff checker
    """
    def __call__(self, in_path, expected_path, out_path):
        assert(in_path.exists())
        assert(expected_path.exists())
        assert(out_path.exists())
        with open('/dev/null', 'w') as null:
            status = subprocess.call(['diff',
                                      expected_path.fullpath,
                                      out_path.fullpath],
                                     stdout=null,
                                     stderr=null)
            return 1.0 if status == 0 else 0.0


@total_ordering
class FilePath(object):
    """Represents a file path to a file. The file may not exists in the
    file system, however the directory where the file resides must exist.
    """
    def __init__(self, arg1, arg2=None):
        """
        Args:
            arg1 (Directory or str): If arg2 is present correspond to
                directory where the file resides. Otherwise it contains
                a full path to file.
            arg2 (str): Basename of the file (including extension).
        """
        if arg2:
            self._directory = arg1
            self._filename = arg2
        else:
            dirname = os.path.dirname(arg1)
            assert(os.path.exists(dirname))
            self._directory = Directory(dirname)
            self._filename = os.path.basename(arg1)

    def __str__(self):
        return self.fullpath

    @property
    def name(self):
        return self._filename

    @property
    def ext(self):
        """str: File extension with beginning dot. """
        return self.splitext()[1]

    @property
    def rootname(self):
        """str: Filename without extension."""
        return self.splitext()[0]

    @property
    def fullpath(self):
        """str: Full path to file"""
        return os.path.join(self._directory.fullpath, self._filename)

    def __eq__(self, other):
        return self.fullath == other.fullpath

    def __lt__(self, other):
        return self.fullpath < other.fullpath

    def chext(self, ext):
        """Change extension of file

        Args:
            ext (string): new extension including dot.

        Returns:
            FilePath: file path with the new extension
        """
        return FilePath(self._directory, self.rootname+ext)

    def splitext(self):
        """Split filename in root name and extension.

        Returns:
            (str, str): A pair where the first component correspond
                to root name and the second to the extension.
        """
        return os.path.splitext(self.fullpath)

    def mtime(self):
        """Returns modification time. If the file does not exists in
        the file system this functions returns the oldest possible date.

        Returns:
            float: Numbers of seconds since the epoch or -inf if the file
                does not exists.
        """
        if self.exists():
            return os.path.getmtime(self.fullpath)
        else:
            return float('-Inf')

    def directory(self):
        """Returns the directory where the file resides.

        Returns:
            Directory
        """
        return self._directory

    def exists(self):
        """Returns true if the file actually exists in the file system

        Returns:
            bool
        """
        return os.path.exists(self.fullpath)

    def __bool__(self):
        return self.exists()


@total_ordering
class Directory(object):
    """Represent a directory in the filesystem. The directory must always
    exists.
    """

    def __init__(self, fullpath):
        """
        Args:
            fullpath (str): Full path to directory.
        """
        assert(os.path.exists(fullpath))
        self._fullpath = os.path.abspath(fullpath)

    @property
    def fullpath(self):
        """str: Full path to directory."""
        return self._fullpath

    def chdir(self, path):
        """Changes directory

        Args:
            path (str)

        Returns:
            Directory:
        """
        return Directory(os.path.join(self.fullpath, path))

    def __eq__(self, other):
        return self.fullath == other.fullpath

    def __lt__(self, other):
        return self.fullpath < other.fullpath

    def lsfile(self, pattern='*'):
        """List files inside this directory. An optional glob pattern
        could be provided. This pattern is concatenated to the full path
        directory.

        Returns:
            List[FilePath]
        """
        pattern = os.path.join(self.fullpath, pattern)
        files = [FilePath(f) for f in glob.glob(pattern) if os.path.isfile(f)]
        return sorted(files)

    def lsdir(self):
        """List directories inside this directory

        Returns:
            List[Directory]
        """
        pattern = os.path.join(self.fullpath, '*')
        dirs = [Directory(f) for f in glob.glob(pattern) if os.path.isdir(f)]
        return sorted(dirs)

    def find_file(self, filename):
        """Finds a file by name in this directory.

        Returns:
            Optional[FilePath]
        """
        return next((f for f in self.lsfile() if filename == f.name), None)


class Binary(object):
    """An executable file."""
    def __init__(self, bin_path):
        """
        Args:
            bin_path (FilePath): Path to the binary.
        """
        assert(bin_path.exists())
        self._bin_path = bin_path

    def __str__(self):
        return self._bin_path.rootname

    def run(self, in_path, out_path, *args):
        """Run binary redirecting standard input and output.

        Args:
            in_path (FilePath): Path to redirect stdin from.
            out_path (FilePath): File to redirecto stdout to.
            *args

        Returns:
            bool: Returns true if the execution terminates with exist code 0
                or false otherwise.
        """
        assert(in_path.exists())
        assert(out_path.exists())
        pid = os.fork()
        if pid == 0:
            with open(in_path.fullpath, 'r') as in_file:
                os.dup2(in_file.fileno(), 0)
            with open(out_path.fullpath, 'w') as out_file:
                os.dup2(out_file.fileno(), 1)
            with open('/dev/null', 'w') as err_file:
                os.dup2(err_file.fileno(), 2)
            os.execl(self._bin_path, self._bin_path, *args)
        (pid, status, rusage) = os.wait4(pid, 0)
        status = os.WEXITSTATUS(status) == 0
        # TODO explanatory status and execution time
        # wtime = rusage.ru_utime + rusage.ru_stime
        return status
