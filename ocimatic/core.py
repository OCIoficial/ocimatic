# coding=UTF-8
import os
import subprocess
import shutil
import re
from contextlib import ExitStack
from math import floor, log

import ocimatic
from ocimatic import ui
from ocimatic.filesystem import FilePath, Directory


class Contest(object):
    """This class represent a contest. A contest is conformed by a list of
    tasks and a titlepage. In ocimatic, a contest is always associated to
    a directory in the filesystem.
    """
    def __init__(self, directory):
        """
        Args:
            directory (Directory): Directory where the contest reside.
        """
        self._directory = directory
        dirs = [d for d in directory.lsdir() if d.find_file('.ocimatic_task')]
        self._tasks = [Task(d,i) for (i,d) in enumerate(dirs)]
        self._titlepage = FilePath(directory, 'titlepage.tex')
        self._compiler = LatexCompiler()

    @staticmethod
    def create_layout(contest_path):
        """Copy contest skeleton to contest_path.

        Args:
            contest_path (str)
        """
        ocimatic_dir = os.path.dirname(__file__)
        shutil.copytree(os.path.join(ocimatic_dir, "resources/contest-skel"),
                        contest_path)

    @property
    def tasks(self):
        """List[Task]"""
        return self._tasks

    @ui.workgroup('Generating Problemset')
    def build_problemset(self):
        """It builds titlepage and statement of all tasks. Then it merges all pdfs
        in a single file.
        """
        self.compile_titlepage()
        for task in self._tasks:
            task.build_statement()
        self.merge_pdfs()

    @ui.work('PDF', 'titlepage.tex')
    def compile_titlepage(self):
        """Compile title page latex
        Returns:
            (bool, msg): Returns status and result message
        """
        st = self._compiler(self._titlepage)
        return (st, 'OK' if st else 'FAILED')

    @ui.work('MERGE', 'problemset.pdf')
    def merge_pdfs(self):
        """Merge statements and title page in a single file """
        pdfs = ' '.join('"%s"' % t.statement.pdf for t in self._tasks
                        if t.statement.pdf)
        titlepage = FilePath(self._directory, 'titlepage.pdf')
        if titlepage:
            pdfs = '"%s" %s' % (titlepage, pdfs)

        cmd = ('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite'
               ' -dPDFSETTINGS=/prepress -sOutputFile=%s %s') % (
                   FilePath(self._directory, 'problemset.pdf'),
                   pdfs
               )
        with FilePath('/dev/null').open('w') as null:
            complete = subprocess.run(cmd,
                                      shell=True,
                                      timeout=20,
                                      # stdin=null)
                                      stdin=null,
                                      stdout=null,
                                      stderr=null)
            st = complete.returncode == 0
            return (st, 'OK' if st else 'FAILED')

    def find_task(self, name):
        """find task with given name.
        Args:
            name (str): Name of the tasks
        Returns:
            Optional[Task]: The task with the given name or None if
                no task with that name is present.
        """
        return next((p for p in self._tasks if p.name == name), None)


class Task(object):
    """This class represent a task. A task consist of a statement,
    a list of correct and partial solutions and a dataset. A task is
    always associated to a directory in the filesystem.

    """
    @staticmethod
    def create_layout(task_path):
        ocimatic_dir = os.path.dirname(__file__)
        shutil.copytree(os.path.join(ocimatic_dir, "resources/task-skel"),
                        task_path)

    def __init__(self, directory, num):
        """
        Args:
            directory (Directory): Directory where the task reside.
            num (int): Position of the task in the problemset starting from 0.
        """
        self._directory = directory

        managers_dir = directory.chdir('managers')

        correct_dir = directory.chdir('solutions/correct')
        self._correct = Solution.get_solutions(correct_dir, managers_dir)
        partial_dir = directory.chdir('solutions/partial')
        self._partial = Solution.get_solutions(partial_dir, managers_dir)

        self._checker = DiffChecker()
        custom_checker = managers_dir.find_file('checker.cpp')
        if custom_checker:
            self._checker = CppChecker(custom_checker)

        self._statement = Statement(directory.chdir('statement'), num)

        self._dataset = DataSet(directory.chdir('dataset'), self._statement)

    @ui.work('ZIP')
    def compress_dataset(self):
        """Compress dataset in a single file data.zip"""
        st = self._dataset.compress()
        return (st, 'OK' if st else 'FAILED')

    @property
    def name(self):
        """str: Name of the task"""
        return self._directory.basename

    def __str__(self):
        return self.name

    @property
    def statement(self):
        """Statement"""
        return self._statement

    @ui.task('Running solutions')
    def run_solutions(self, partial=False):
        """Run all solutions and report outcome and running time

        Args:
            partial (bool): If true it runs partial solutions as well.
        """
        solutions = self._correct
        if partial:
            solutions.extend(self._partial)

        for sol in solutions:
            sol.run(self._dataset, self._checker)

    @ui.task('Checking dataset')
    def check_dataset(self):
        """Check if dataset input/output is correct by running all correct
        solutions.
        """
        for sol in self._correct:
            sol.run(self._dataset, self._checker, check=True, sample=True)

    @ui.task('Building solutions')
    def build_solutions(self):
        """Force a rebuilding of all solutions, both partial and corrects."""
        for sol in self._correct + self._partial:
            sol.build()

    @ui.task('Generating expected output')
    def gen_expected(self, sample=False):
        """Generate expected outputs files for dataset by running one of the
        correct solutions.
        """
        if not self._correct:
            ui.fatal_error('No correct solution.')
        self._correct[0].gen_expected(self._dataset, sample=sample)

    def build_statement(self):
        """Generate pdf for the statement"""
        return self._statement.build()


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

    @ui.workgroup()
    def run(self, dataset, checker, check=False, sample=False):
        """Run this solution for all test in a dataset.
        Args:
            dataset (Dataset)
            checker (Checker): Checker to compute outcome.
            check  (bool): If true this only report if expected output
                correspond to solution output.
            sample (bool): If true run solution with sample test data from
                statement.
        """
        dataset.run(self.binary, checker, check=check, sample=sample)

    @ui.workgroup()
    def gen_expected(self, dataset, sample=False):
        """Generate expected output files for all test cases in the dataset
        running this solution.
        Args:
            dataset (Dataset)
            sample (bool): If true expected output file for are generated for
                sample test data from statement.
        """
        dataset.gen_expected(self.binary, sample=sample)

    @ui.work('Build')
    def build(self):
        """Build solution.
        Returns:
            (bool, str): A tuple containing status and result message.
        """
        st = self._build()
        msg = 'OK' if st else 'FAILED'
        return (st, msg)

    @property
    def binary(self):
        """Optional[Binary]: Binary file of this solution"""
        if self._bin_path.mtime() < self._source.mtime():
            if not self.build():
                return None
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
        self._compiler = CppCompiler(['-I"%s"' % managers])
        self._grader = managers.find_file('grader.cpp')
        self._bin_path = self._source.chext('.bin')

    def __str__(self):
        return self._source.path

    def _build(self):
        """Compile solution with a CppCompiler. Solutions is compiled with a
        grader if present.
        """
        sources = [self._source]
        if self._grader:
            sources.append(self._grader)
        return self._compiler(sources, self._bin_path)


class CppCompiler(object):
    """Compiles C++ code
    """
    _flags = ['-std=c++11', '-O2']
    def __init__(self, flags=[]):
        flags = list(set(self._flags+flags))
        self._cmd_template = 'g++ %s -o %%s %%s' % ' '.join(flags)

    def __call__(self, sources, out):
        """Compiles a list of C++ sources.

        Args:
            sources (List[FilePath]|FilePath): Source or list of sources.
            out (FilePath): Output path for binary

        Returns:
            bool: True if compilations succeed, False otherwise.
        """
        out = '"%s"' % out
        sources = [sources] if isinstance(sources, FilePath) else sources
        sources = ' '.join('"%s"' % w for w in sources)
        cmd = self._cmd_template % (out, sources)

        with FilePath('/dev/null').open('w') as null:
            complete = subprocess.run(cmd,
                                      stdout=null,
                                      stderr=null,
                                      shell=True)
            return complete.returncode == 0


class DataSet(object):
    """Test data"""
    def __init__(self, directory, statement):
        """
        Args:
            directory (Directory): dataset directory.
            statement (Statement): statement to look for sample test data.
        """
        self._directory = directory
        self._tests = [Test(f) for f in directory.lsfile('*/*.in')]
        self._samples = statement.io_samples();

    # TODO error handling
    def run(self, binary, checker, check=False, sample=False):
        """Run binary with all test in this dataset as input and
        check outcome of input.

        Args:
            binary (Binary)
            checker (Checker): Checker to check outcome
            check  (bool): If true this only report if expected output
                correspond to binary execution output.
            sample (bool): If true run solution with sample test data.
        """
        for test in self._tests:
            test.run(binary, checker, check=check);
        if sample:
            for test in self._samples:
                test.run(binary, checker, check=check);

    def gen_expected(self, binary, sample=False):
        """Run binary with all test in this dataset as input to generate expected
        output files.
        Args:
            binary (Binary)
            sample (bool): If true run binary with sample data as input as well.
        """
        for test in self._tests:
            test.gen_expected(binary)
        if sample:
            for test in self._samples:
                test.gen_expected(binary)

    def compress(self):
        """Compress all test in this dataset in a single zip file. This function
        prepend the subdirectory basename to all files. The order in which
        subdirectories appears in the dataset directory is relevant, so files
        maintains this order.
        """
        tmpdir = Directory.tmpdir()
        w = floor(log(len(self._tests), 10)) + 1
        in_format = "%%s-%%0%dd.in" % w
        sol_format = "%%s-%%0%dd.sol" % w
        for (i, test) in enumerate(self._tests):
            if test.expected_path:
                in_name = in_format % (test.directory().basename, i)
                sol_name = sol_format % (test.directory().basename, i)
                test.in_path.copy(FilePath(tmpdir, in_name))
                test.expected_path.copy(FilePath(tmpdir, sol_name))
        cmd = "cd %s && zip data.zip *.in *.sol" % tmpdir
        with FilePath('/dev/null').open('a') as null:
            complete = subprocess.run(cmd, stdout=null, shell=True)
            dst_file = FilePath(self._directory, 'data.zip')
            FilePath(tmpdir, 'data.zip').copy(dst_file)
            tmpdir.rmtree()

            return complete.returncode == 0


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
    def directory(self):
        """Diretory: diretory where this test reside"""
        return self._in_path.directory

    @ui.work('Gen')
    def gen_expected(self, binary):
        """Run binary with this test as input to generate expected output file
        Args:
            binary (Binary)
        Returns:
            (bool, msg): A tuple containing status and result message.
        """
        (st, _, errmsg) = binary.run(self.in_path, self.expected_path)
        msg = 'OK' if st else errmsg
        return (st, msg)


    @ui.work('Run')
    def run(self, binary, checker, check=False):
        """Run binary whit this test as input and check output correcteness
        Args:
            binary (Binary)
            checker (Checker): Checker to check outcome
            check  (bool): If true this only report if expected output
                correspond to binary execution output.
        """
        out_path = FilePath.tmpfile()
        if self.expected_path:
            (st, time, errmsg) = binary.run(self.in_path, out_path)

            # Execution failed
            if not st:
                if check:
                    return (st, errmsg)
                else:
                    return (st, '%s [%.2fs]' % (errmsg, time))

            outcome = checker(self.in_path,
                              self.expected_path,
                              out_path)
            if check:
                msg = 'OK' if outcome == 1.0 else 'FAILED'
                st = outcome == 1.0
                return (st, msg)
            else:
                return (st, '%s [%.2fs]' % (outcome, time))
        else:
            out_path.remove()
            return (False, 'No expected output file')

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
        """Check outcome.

        Args:
            in_path (FilePath): Input file.
            expected_path (FilePath): Expected solution file
            out_path (FilePath): Output file.

        Returns:
            float: Float between 0.0 and 1.0 indicating result.
        """
        NotImplementedError("Class %s doesn't implement __call__()" % (
            self.__class__.__name__))


class DiffChecker(Checker):
    """White diff checker
    """
    def __call__(self, in_path, expected_path, out_path):
        """Performs a white diff between expected output and output files
        Parameters correspond to convention for checker in cms.
        Args:
            in_path (FilePath)
            expected_path (FilePath)
            out_path (FilePath)
        """
        assert(in_path.exists())
        assert(expected_path.exists())
        assert(out_path.exists())
        with FilePath('/dev/null').open('w') as null:
            complete = subprocess.run(['diff',
                                       expected_path.path,
                                       out_path.path],
                                      stdout=null,
                                      stderr=null)
            return 1.0 if complete.returncode == 0 else 0.0

class CppChecker(Checker):
    def __init__(self, source):
        """
        Args:
            source (FilePath)
        """
        self._source = source
        self._compiler = CppCompiler(['-I"%s"' % source.directory()])
        self._binary_path = FilePath(source.directory(), 'checker')

    def __call__(self, in_path, expected_path, out_path):
        """Run checker to evaluate outcome. Parameters correspond to convention
        for checker in cms.
        Args:
            in_path (FilePath)
            expected_path (FilePath)
            out_path (FilePath)
        """
        assert(in_path.exists())
        assert(expected_path.exists())
        assert(out_path.exists())
        if self._binary_path.mtime() < self._source.mtime():
            if not self.build():
                return 0.0
        with FilePath('/dev/null').open('w') as null:
            complete = subprocess.run([self._binary_path.path,
                                       in_path.path,
                                       expected_path.path,
                                       out_path.path],
                                      universal_newlines=True,
                                      stdout=subprocess.PIPE,
                                      stderr=null)
            if complete.returncode == 0:
                return float(complete.stdout)

            return 0.0

    def build(self):
        """Build source of the checker
        Returns:
            bool: True if compilation is successful. False otherwise
        """
        return self._compiler(self._source, self._binary_path)


from ctypes import Structure, c_long, CDLL, c_int, POINTER, byref
from ctypes.util import find_library
CLOCK_MONOTONIC = 1

class timespec(Structure):
    _fields_ = [
        ('tv_sec', c_long),
        ('tv_nsec', c_long)
        ]
librt_filename = find_library('rt')
if not librt_filename:
    # On Debian Lenny (Python 2.5.2), find_library() is unable
    # to locate /lib/librt.so.1
    librt_filename = 'librt.so.1'
librt = CDLL(librt_filename)
_clock_gettime = librt.clock_gettime
_clock_gettime.argtypes = (c_int, POINTER(timespec))
def monotonic_time():
    t = timespec()
    _clock_gettime(CLOCK_MONOTONIC, byref(t))
    return t.tv_sec + t.tv_nsec / 1e9


class Binary(object):
    """An entity that may be executed redirecting stdin and stdout to specific
    files.
    """

    signal = {
        1: 'SIGHUP', 2: 'SIGINT', 3: 'SIGQUIT', 4: 'SIGILL', 5: 'SIGTRAP',
        6: 'SIGABRT', 7: 'SIGEMT', 8: 'SIGFPE', 9: 'SIGKILL', 10: 'SIGBUS',
        11: 'SIGSEGV', 12: 'SIGSYS', 13: 'SIGPIPE', 14: 'SIGALRM', 15: 'SIGTERM',
        16: 'SIGURG', 17: 'SIGSTOP', 18: 'SIGTSTP', 19: 'SIGCONT', 20: 'SIGCHLD',
        21: 'SIGTTIN', 22: 'SIGTTOU', 23: 'SIGIO', 24: 'SIGXCPU', 25: 'SIGXFSZ',
        26: 'SIGVTALRM', 27: 'SIGPROF', 28: 'SIGWINCH', 29: 'SIGINFO',
        30: 'SIGUSR1', 31: 'SIGUSR2',
    }
    def __init__(self, bin_path):
        """
        Args:
            bin_path (FilePath): Path to the binary.
        """
        assert(bin_path.exists())
        self._bin_path = bin_path

    def __str__(self):
        return self._bin_path.name

    def run(self, in_path, out_path, *args):
        """Run binary redirecting standard input and output.

        Args:
            in_path (FilePath): Path to redirect stdin from.
            out_path (FilePath): File to redirecto stdout to.

        Returns:
            (bool, str, float): Returns a tuple (status, time, errmsg).
                status is True if the execution terminates with exit code zero
                or False otherwise.
                time corresponds to execution time.
                if status is False errmsg contains an error message.
        """
        assert(in_path.exists())
        with ExitStack() as stack:
            in_file = stack.enter_context(in_path.open('r'))
            out_file = stack.enter_context(out_path.open('w'))
            null = stack.enter_context(open('/dev/null', 'w'))

            start = monotonic_time()
            try:
                complete = subprocess.run([self._bin_path.path],
                                          timeout=ocimatic.config['timeout'],
                                          stdin=in_file,
                                          stdout=out_file,
                                          stderr=null)
            except subprocess.TimeoutExpired as to:
                return (False, monotonic_time() - start, 'Execution timed out')
            time = monotonic_time() - start
            status = complete.returncode == 0
            msg = ''
            if not status:
                sig = -complete.returncode
                msg = 'Execution killed with signal %d: %s' % (
                    sig, self.signal[sig])
            return (status, time, msg)


class Statement(object):
    """Represent a statement. A statement is formed by a latex source and a pdf
    file.
    """
    def __init__(self, directory, num):
        """
        Args:
            directory (Directory): Directory to search for statement source file.
            num (int): Number of the statement in the contest starting from 0
        """
        assert(FilePath(directory, 'statement.tex'))
        self._source = FilePath(directory, 'statement.tex')
        self._pdf = self._source.chext('.pdf')
        self._compiler = LatexCompiler()
        self._directory = directory
        self._num = num

    @property
    def pdf(self):
        """Returns path to pdf file and compiles it if necessary.
        Returns:
            Optional[FilePath]: The file path if the binary is present or None
                if pdf file cannot be generated.
        """
        if self._pdf.mtime() < self._source.mtime():
            st = self.build()
            if not st:
                return None
        return self._pdf

    def __str__(self):
        return self._source.path

    @ui.work('PDF')
    def build(self):
        """Compile statement latex source
        Returns:
           (bool, msg) a tuple containing status code and result message.

        """
        os.environ['OCIMATIC_PROBLEM_NUMBER'] = chr(ord('A')+self._num)
        st = self._compiler(self._source)
        return (st, 'OK' if st else 'FAILED')

    def io_samples(self):
        """Find sample input data in the satement
        Returns:
            List[Test]: list of tests
        """
        latex_file = self._source.open('r')
        samples = set()
        for line in latex_file:
            m = re.match(r'[^%]*\\sampleIO{([^}]*)}', line)
            m and samples.add(m.group(1))
        latex_file.close()
        samples = [Test(FilePath(self._directory, s+'.in')) for s in samples]
        return samples



class LatexCompiler(object):
    """Compiles latex source"""
    def __init__(self, cmd='pdflatex',
                 flags=['--shell-escape', '-interaction=batchmode']):
                 # flags=['--shell-escape']):
        """
        Args:
            cmd (str): command to compile files. default to pdflatex
            flags (List[str]): list of flags to pass to command
        """
        self._cmd = cmd
        self._flags = flags

    def __call__(self, source):
        """It compiles a latex source leaving the pdf in the same diretory of
        the source.
        Args:
            source (FilePath): path of file to compile
        """
        flags = ' '.join(self._flags)
        cmd = 'cd %s && %s %s %s' % (
            source.directory(), self._cmd, flags, source.name)
        with FilePath('/dev/null').open('w') as null:
            complete = subprocess.run(cmd,
                                      shell=True,
                                      # stdin=null)
                                      stdin=null,
                                      stdout=null,
                                      stderr=null)
            return complete.returncode == 0

