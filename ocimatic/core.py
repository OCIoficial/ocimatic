# coding=UTF-8
import os
import subprocess
import re
import shutil
from contextlib import ExitStack
from distutils.dir_util import copy_tree

import ocimatic
from ocimatic import ui
from ocimatic.filesystem import FilePath, Directory


class Contest(object):
    """This class represents a contest. A contest is formed by a list of
    tasks and a titlepage. A contest is associated to a directory in the
    filesystem.
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
        """Copies contest skeleton to contest_path.

        Args:
            contest_path (str)
        """
        ocimatic_dir = os.path.dirname(__file__)
        copy_tree(os.path.join(ocimatic_dir, "resources/contest-skel"),
                  contest_path)

    @property
    def tasks(self):
        """List[Task]"""
        return self._tasks


    @ui.supergroup('Generating problemset')
    def build_problemset(self):
        """It builds the titlepage and the statement of all tasks. Then it merges
        all pdfs in a single file.
        """
        self.build_problemset_twoside()
        self.build_problemset_oneside()

    @ui.workgroup('oneside')
    def build_problemset_oneside(self):
        os.environ['OCIMATIC_SIDENESS'] = 'oneside'
        self.compile_titlepage()

        for (i, task) in enumerate(self._tasks):
            task.build_statement()
        self.merge_pdfs('oneside.pdf')

    @ui.workgroup('twoside')
    def build_problemset_twoside(self):
        os.environ['OCIMATIC_SIDENESS'] = 'twoside'
        self.compile_titlepage()

        for (i, task) in enumerate(self._tasks):
            last = i == len(self._tasks) - 1
            blank_page = last and ocimatic.config['last_blank_page']
            task.build_statement(blank_page=blank_page)
        self.merge_pdfs('twoside.pdf')

    @ui.supergroup('Compressing contest')
    def compress(self):
        """Compress statement and dataset of all tasks in a single file"""
        tmpdir = Directory.tmpdir()
        try:
            for task in self._tasks:
                task.copy_to(tmpdir)
            cmd = 'cd %s && zip -r contest.zip .' % tmpdir
            st = subprocess.call(cmd, stdout=subprocess.DEVNULL, shell=True)
            contest = FilePath(self._directory, '%s.zip' % self.name)
            FilePath(tmpdir, 'contest.zip').copy(contest)
        finally:
            tmpdir.rmtree()

        return st == 0

    @ui.isolated_work('PDF', 'titlepage.tex')
    def compile_titlepage(self):
        """Compile title page latex
        Returns:
            (bool, msg): Status and result message
        """
        st = self._compiler(self._titlepage)
        return (st, 'OK' if st else 'FAILED')

    @ui.isolated_work('MERGE')
    def merge_pdfs(self, filename):
        """Merges statements and title page in a single file """
        if not shutil.which('gs'):
            return (False, 'Cannot find gs')

        pdfs = ' '.join('"%s"' % t.statement.pdf for t in self._tasks
                        if t.statement.pdf)
        titlepage = FilePath(self._directory, 'titlepage.pdf')
        if titlepage.exists():
            pdfs = '"%s" %s' % (titlepage, pdfs)

        cmd = ('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite'
               ' -dPDFSETTINGS=/prepress -sOutputFile=%s %s') % (
                   FilePath(self._directory, filename),
                   pdfs
               )
        complete = subprocess.run(cmd,
                                    shell=True,
                                    timeout=20,
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        st = complete.returncode == 0
        return (st, 'OK' if st else 'FAILED')

    @property
    def name(self):
        """str: Name of the contest"""
        return self._directory.basename

    def __str__(self):
        return self.name

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
    """This class represents a task. A task consists of a statement,
    a list of correct and partial solutions and a dataset. A task is
    associated to a directory in the filesystem.
    """
    @staticmethod
    def create_layout(task_path):
        ocimatic_dir = os.path.dirname(__file__)
        copy_tree(os.path.join(ocimatic_dir, "resources/task-skel"),
                  task_path, preserve_symlinks=1)

    def __init__(self, directory, num):
        """
        Args:
            directory (Directory): Directory where the task resides.
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

        self._dataset = Dataset(directory.chdir('dataset'), self._statement)

        self._dataset_plan = DatasetPlan(directory.chdir('attic'),
                                         directory,
                                         directory.chdir('dataset'))

    def copy_to(self, directory):
        new_dir = Directory.create(directory, str(self))

        (st, msg) = self.compress_dataset()
        if st:
            dataset = FilePath(new_dir, 'data.zip')
            FilePath(self._directory.chdir('dataset'), 'data.zip').copy(dataset)

        (st, msg) = self.build_statement()
        if st:
            statement = FilePath(new_dir, 'statement.pdf')
            FilePath(self._directory.chdir('statement'), 'statement.pdf').copy(statement)

    @ui.task('Generating dataset')
    def gen_input(self):
        self._dataset_plan.run()

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

    def solutions(self, partial=False):
        for solution in self._correct:
            yield solution
        if partial:
            for solution in self._partial:
                yield solution

    @property
    def statement(self):
        """Statement"""
        return self._statement

    @ui.task('Normalizing')
    def normalize(self):
        self._dataset.normalize()

    @ui.task('Running solutions')
    def run_solutions(self, partial=False, pattern=None):
        """Run all solutions reporting outcome and running time.

        Args:
            partial (bool): If true it runs partial solutions as well.
            pattern (Optional[string]): If present it only runs the solutions that
                contain pattern as substring.
        """
        print(pattern)
        for sol in self.solutions(partial):
            if not pattern or pattern in sol.name:
                sol.run(self._dataset, self._checker)

    @ui.task('Checking dataset')
    def check_dataset(self):
        """Check if dataset input/output is correct by running all correct
        solutions.
        """
        for sol in self.solutions():
            sol.run(self._dataset, self._checker, check=True, sample=True)

    @ui.task('Building solutions')
    def build_solutions(self, pattern=None):
        """Forces a rebuilding of all solutions, both partial and corrects."""
        for sol in self.solutions(partial=True):
            if pattern is not None and pattern in sol.name:
                sol.build()

    @ui.task('Generating expected output')
    def gen_expected(self, sample=False):
        """Generate expected outputs files for dataset by running one of the
        correct solutions.
        """
        if not self._correct:
            ui.fatal_error('No correct solution.')
        self._correct[0].gen_expected(self._dataset, sample=sample)

    def build_statement(self, blank_page=False):
        """Generate pdf for the statement"""
        return self._statement.build(blank_page=blank_page)


class Solution(object):
    """Abstract class to represent a solution
    """
    @staticmethod
    def get_solutions(solutions_dir, managers_dir):
        """Search for solutions in a directory.

        Args:
            solutions_dir (Directory): Directory to look for solutions.
            managers_dir (Directory): Directory where managers reside.
                This is used to provide necessary files for compilation,
                for example, when solutions are compiled with a grader.

        Returns:
            List[Solution]: List of solutions.
        """
        solutions = []
        for f in solutions_dir.lsfile():
            if f.ext == CppSolution.ext:
                solutions.append(CppSolution(f, managers_dir))
            if f.ext == JavaSolution.ext:
                solutions.append(JavaSolution(f, managers_dir))
        return solutions

    @ui.supergroup()
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
        runnable = self.get_and_build()
        if runnable:
            dataset.run(runnable, checker, sample=sample, check=check)

    @ui.workgroup()
    def gen_expected(self, dataset, sample=False):
        """Generate expected output files for all test cases in the given dataset
        running this solution.
        Args:
            dataset (Dataset)
            sample (bool): If true expected output file for are generated for
                sample test data from statement.
        """
        runnable = self.get_and_build()
        if runnable:
            dataset.gen_expected(runnable, sample=sample)

    @ui.work('Build')
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
            (st, msg) = self.build()
            if not st:
                return None
        return self.get_runnable()

    def get_runnable(self):
        raise NotImplementedError("Class %s doesn't implement get_runnable()" % (
            self.__class__.__name__))

    def build_time(self):
        raise NotImplementedError("Class %s doesn't implement build_time()" % (
            self.__class__.__name__))

    @property
    def name(self):
        return self._source.name


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

    def get_runnable(self):
        return Runnable(self._bin_path.path)

    def build_time(self):
        return self._bin_path.mtime()

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

        complete = subprocess.run(cmd,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    shell=True)
        return complete.returncode == 0

class JavaSolution(Solution):
    """Solution written in C++. This solutions is compiled with
    a grader if one is present in the managers directory.
    """
    ext = '.java'

    def __init__(self, source, managers):
        """
        Args:
            source (FilePath): Source code.
            managers (Directory): Directory where managers reside.
        """
        assert(source.ext == self.ext)
        self._source = source
        self._compiler = JavaCompiler()
        # self._grader = managers.find_file('grader.cpp')
        self._classname = self._source.rootname()
        self._classpath = self._source.directory().path
        self._bytecode = self._source.chext('.class')

    def get_runnable(self):
        return Runnable('java', ['-cp', self._classpath,
                                 self._classname])

    def build_time(self):
        return self._bytecode.mtime()

    def __str__(self):
        return self._source.path

    def _build(self):
        """Compile solution with the JavaCompiler.
        @TODO (NL: 26/09/2016) Compile solutions with a grader if present.
        """
        sources = [self._source]
        # if self._grader:
        #     sources.append(self._grader)
        return self._compiler(sources)


class JavaCompiler(object):
    """Compiles Java code
    """
    _flags = []
    def __init__(self, flags=[]):
        flags = list(set(self._flags+flags))
        self._cmd_template = 'javac %s %%s' % ' '.join(flags)

    def __call__(self, sources):
        """Compiles a list of Java sources.

        Args:
            sources (List[FilePath]|FilePath): Source or list of sources.
            out (FilePath): Output path for bytecode

        Returns:
            bool: True if compilation succeed, False otherwise.
        """
        sources = [sources] if isinstance(sources, FilePath) else sources
        sources = ' '.join('"%s"' % w for w in sources)
        cmd = self._cmd_template % sources

        complete = subprocess.run(cmd,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  shell=True)
        return complete.returncode == 0


# TODO refactor statement out of dataset
class Dataset(object):
    """Test data"""
    def __init__(self, directory, statement=None,
                 in_ext='.in', sol_ext='.sol'):
        """
        Args:
            directory (Directory): dataset directory.
            statement (Optional[Statement]): optional statement to look for
                sample test data.
        """
        self._directory = directory
        self._in_ext = in_ext
        self._sol_ext = sol_ext
        self._subtasks = [Subtask(d, in_ext, sol_ext) for d in directory.lsdir()]
        self._sampledata = SampleData(statement, in_ext=in_ext, sol_ext=sol_ext)

    def gen_expected(self, runnable, sample=False):
        for subtask in self._subtasks:
            subtask.gen_expected(runnable)
        if sample:
            self._sampledata.gen_expected(runnable)

    def run(self, runnable, checker, check=False, sample=False):
        for subtask in self._subtasks:
            subtask.run(runnable, checker, check=check)
        if sample:
            self._sampledata.run(runnable, checker, check=check)

    def compress(self):
        """Compress all test cases in this dataset in a single zip file.
        The basename of the corresponding subtask subdirectory is prepended
        to each file.
        """
        tmpdir = Directory.tmpdir()

        try:
            copied = 0
            for subtask in self._subtasks:
                copied += subtask.copy_to(tmpdir)

            if not copied:
                ui.show_message("Warning", "no files in dataset", ui.WARNING)
                return

            cmd = 'cd %s && zip data.zip *%s *%s' % (tmpdir,
                                                    self._in_ext,
                                                    self._sol_ext)
            st = subprocess.call(cmd, stdout=subprocess.DEVNULL, shell=True)
            dst_file = FilePath(self._directory, 'data.zip')
            FilePath(tmpdir, 'data.zip').copy(dst_file)
        finally:
            tmpdir.rmtree()

        return st == 0

    def normalize(self):
        for subtask in self._subtasks:
            subtask.normalize()
        self._sampledata.normalize()


class Subtask(object):
    def __init__(self, directory, in_ext='.in', sol_ext='.sol'):
        self._tests = []
        self._in_ext = in_ext
        self._sol_ext = sol_ext
        for f in directory.lsfile('*'+self._in_ext):
            self._tests.append(Test(f, f.chext(sol_ext)))
        self._name = directory.basename

    def copy_to(self, directory):
        copied = 0
        for test in self._tests:
            if test.expected_path:
                in_name = "%s-%s" % (self._name, test.in_path.name)
                sol_name = "%s-%s" % (self._name, test.expected_path.name)
                test.in_path.copy(FilePath(directory, in_name))
                test.expected_path.copy(FilePath(directory, sol_name))
                copied += 1
        return copied

    def normalize(self):
        for test in self._tests:
            test.normalize()

    @ui.workgroup()
    def run(self, runnable, checker, check=False):
        for test in self._tests:
            test.run(runnable, checker, check=check)

    @ui.workgroup()
    def gen_expected(self, runnable):
        for test in self._tests:
            test.gen_expected(runnable)

    def __str__(self):
        return self._name


class SampleData(Subtask):
    def __init__(self, statement, in_ext='.in', sol_ext='.sol'):
        self._in_ext = in_ext
        self._sol_ext = sol_ext
        tests = statement.io_samples() if statement else []
        self._tests = [Test(f.chext(in_ext), f.chext(sol_ext)) for f in tests]

    def __str__(self):
        return 'Sample'


class Test(object):
    """A single test file. Expected output file may not exist"""
    def __init__(self, in_path, expected_path):
        """
        Args:
            in_path (FilePath)
            expected_path (FilePath)
        """
        assert(in_path.exists())
        self._in_path = in_path
        self._expected_path = expected_path

    def __str__(self):
        return str(self._in_path)

    @property
    def directory(self):
        """Diretory: diretory where this test reside"""
        return self._in_path.directory

    @ui.work('Gen')
    def gen_expected(self, runnable):
        """Run binary with this test as input to generate expected output file
        Args:
            runnable (Runnable)
        Returns:
            (bool, msg): A tuple containing status and result message.
        """
        (st, _, errmsg) = runnable.run(self.in_path, self.expected_path)
        msg = 'OK' if st else errmsg
        return (st, msg)

    @ui.work('Run')
    def run(self, runnable, checker, check=False):
        """Run runnable whit this test as input and check output correctness
        Args:
            runnable (Runnable)
            checker (Checker): Checker to check outcome
            check  (bool): If true this only report if expected output
                correspond to binary execution output.
        """
        out_path = FilePath.tmpfile()
        if self.expected_path:
            (st, time, errmsg) = runnable.run(self.in_path, out_path)

            # Execution failed
            if not st:
                if check:
                    return (st, errmsg)
                else:
                    # return (st, '%s [%.2fs]' % (errmsg, time))
                    return (st, '%s' % errmsg)

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

    @ui.work('Normalize')
    def normalize(self):
        if not shutil.which('dos2unix'):
            return (False, 'Cannot find dos2unix')
        if not shutil.which('sed'):
            return (False, 'Cannot find sed')
        tounix_input = 'dos2unix "%s"' % self._in_path
        tounix_expected = 'dos2unix "%s"' % self._expected_path
        sed_input = "sed -i -e '$a\\' \"%s\"" % self._in_path
        sed_expected = "sed -i -e '$a\\' \"%s\"" % self._expected_path
        null = subprocess.DEVNULL
        st = subprocess.call(tounix_input, stdout=null, stderr=null, shell=True)
        st += subprocess.call(sed_input, stdout=null, stderr=null, shell=True)
        if self.expected_path:
            st += subprocess.call(tounix_expected, stdout=null, stderr=null, shell=True)
            st += subprocess.call(sed_expected, stdout=null, stderr=null, shell=True)
        return (st == 0, 'OK' if st == 0 else 'FAILED')


class DatasetPlan(object):
    """Functionality to read and run a plan for generating dataset."""

    def __init__(self, directory, task_directory, dataset_directory):
        self._directory = directory
        self._task_directory = task_directory
        self._dataset_directory = dataset_directory
        self._cpp_compiler = CppCompiler()


    def run(self, name='testplan.txt'):
        path = FilePath(self._directory, name)
        if not path.exists():
            ui.fatal_error('No such file plan for creating dataset: "%s"' % path)
        (subtasks, cmds) = self.parse_file(path)

        for st in range(1, subtasks + 1):
            dire = FilePath(self._dataset_directory, 'st%d' % st).get_or_create_dir()
            dire.clear()

        if not cmds:
            ui.show_message("Warning", 'no commands were executed for the plan.',
                            ui.WARNING)

        for (st, tests) in sorted(cmds.items()):
            self.run_subtask('Subtask %d'%st, st, tests)

    @ui.isolated_workgroup()
    def run_subtask(self, msg, st, tests):
        for test in tests:
            st_dir = FilePath(self._dataset_directory, 'st%d' % st).get_or_create_dir()
            cmd = test['cmd']
            if cmd == 'copy':
                self.copy(test['file'], st_dir)
            else:
                if cmd in ['cpp', 'py']:
                    source = FilePath(self._directory, test['source'])
                    test_file = FilePath(st_dir,
                                            test['source']).chext('-%d.in' % test['num'])
                    if cmd == 'cpp':
                        self.run_cpp_generator(source, test['args'], test_file)
                    elif cmd == 'py':
                        self.run_py_generator(source, test['args'], test_file)
                elif cmd == 'run':
                    bin_path = FilePath(self._directory, test['bin'])
                    test_file = FilePath(st_dir, '%s-%s.in' % (test['bin'], test['num']))
                    self.run_bin_generator(bin_path, test['args'], test_file)
                else:
                    ui.fatal_error('unexpected command when running plan: %s '
                                    % cmd)

    @ui.isolated_work('Copy')
    def copy(self, src, dst):
        fp = FilePath(self._task_directory, src)
        if not fp.exists():
            return (False, 'No such file')
        try:
            fp.copy(dst)
            return (True, 'OK')
        except:
            return (False, 'Unexpected error when copying file')

    @ui.isolated_work('Gen')
    def run_cpp_generator(self, source, args, dst):
        if not source.exists():
            return (False, 'No such file')
        binary = source.chext('.bin')
        if binary.mtime() < source.mtime():
            st = self._cpp_compiler(source, binary)
            if not st:
                return (st, 'Failed to build generator')

        (st, time, msg) = Runnable(binary.path).run(None, dst, args)
        return (st, msg)

    @ui.isolated_work('Gen')
    def run_py_generator(self, source, args, dst):
        if not source.exists():
            return (False, 'No such file')
        (st, time, msg) = Runnable('python').run(None, dst, [source.path]+args)
        return (st, msg)

    @ui.isolated_work('Gen')
    def run_bin_generator(self, bin_path, args, dst):
        if not bin_path.exists():
            return (False, 'No such file')
        if not Runnable.is_callable(bin_path):
            return (False, 'Cannot run file, it may not have correct permissions')
        (st, time, msg) = Runnable(bin_path.path).run(None, dst, args)
        return (st, msg)


    def parse_file(self, path):
        """
        Args:
            path (FilePath)
        """
        cmds = {}
        st = 0
        test = 1
        for (lineno, line) in enumerate(path.open('r').readlines(), 1):
            line = line.strip()
            subtask_header = re.compile(r'\s*\[\s*Subtask\s*(\d+)\s*\]\s*')

            if not line:
                continue
            if line[0] != '#':
                match = subtask_header.fullmatch(line)
                if match:
                    found_st = int(match.group(1))
                    if st + 1 != found_st:
                        fatal_error(
                            'line %d: found subtask %d, but subtask %d was expected' %
                            (lineno, found_st, st)
                        )
                    st += 1
                    cmds[st] = []
                    test = 1
                else:
                    if st == 0:
                        ui.fatal_error('line %d: found command before declaring a subtask.' % lineno)

                    args = line.split()

                    if args[0] == 'copy':
                        if len(args) > 2:
                            fatal_error('line %d: command copy expects exactly one argument.' % lineno)
                        cmds[st].append({
                            'cmd': 'copy',
                            'num': test,
                            'file': args[1]
                        })
                    else:
                        f = FilePath(self._directory, args[0])
                        name = '%s-%s' % (st, test)
                        if f.ext == '.cpp':
                            cmds[st].append({
                                'cmd': 'cpp',
                                'num': test,
                                'source': args[0],
                                'args': args[1:]
                            })
                        elif f.ext == '.py':
                            cmds[st].append({
                                'cmd': 'py',
                                'num': test,
                                'source': args[0],
                                'args': args[1:]
                            })
                        else:
                            cmds[st].append({
                                'cmd': 'run',
                                'num': test,
                                'bin': args[0],
                                'args': args[1:]
                            })
                        # else:
                        #     ui.fatal_error('line %d: unexpected command `%s` when'
                        #                    ' parsing plan' % (lineno, args[0]))
                    test += 1
        return (st, cmds)


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


class Runnable(object):
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

    @staticmethod
    def is_callable(file_path):
        return shutil.which(file_path.path) is not None

    def __init__(self, command, args=[]):
        """
        Args:
            bin_path (string): Command to execute.
            args (List[string]): List of arguments to pass to the command.
        """
        assert(shutil.which(command))
        self._cmd = [command] + args

    def __str__(self):
        return self._cmd[0]

    def run(self, in_path, out_path, args=[]):
        """Run binary redirecting standard input and output.

        Args:
            in_path (Optional[FilePath]): Path to redirect stdin from. If None
                input is redirected from /dev/null.
            out_path (Optional[FilePath]): File to redirec stdout to. If None
                output is redirected to /dev/null.
            args (List[str]): Additional parameters

        Returns:
            (bool, str, float): Returns a tuple (status, time, errmsg).
                status is True if the execution terminates with exit code zero
                or False otherwise.
                time corresponds to execution time.
                if status is False errmsg contains an explanatory error
                message, otherwise it contains a success message.
        """
        assert(in_path is None or in_path.exists())
        with ExitStack() as stack:
            if in_path is None:
                in_path = FilePath('/dev/null')
            in_file = stack.enter_context(in_path.open('r'))
            if not out_path:
                out_path = FilePath('/dev/null')
            out_file = stack.enter_context(out_path.open('w'))

            start = monotonic_time()
            self._cmd.extend(args)
            try:
                complete = subprocess.run(self._cmd,
                                          timeout=ocimatic.config['timeout'],
                                          stdin=in_file,
                                          stdout=out_file,
                                          universal_newlines=True,
                                          stderr=subprocess.PIPE)
            except subprocess.TimeoutExpired as to:
                return (False, monotonic_time() - start, 'Execution timed out')
            time = monotonic_time() - start
            ret = complete.returncode
            status = ret == 0
            msg = 'OK'
            if not status:
                stderr = complete.stderr.strip('\n')
                if 0 < len(stderr) < 75:
                    msg = stderr
                else:
                    if ret < 0:
                        sig = -ret
                        msg = 'Execution killed with signal %d' % sig
                        if sig in self.signal:
                            msg += ': %s' % self.signal[sig]
                    else:
                        msg = 'Execution ended with error (return code %d)' % ret
            return (status, time, msg)


class Statement(object):
    """Represents a statement. A statement is formed by a latex source and a pdf
    file.
    """
    def __init__(self, directory, num):
        """
        Args:
            directory (Directory): Directory to search for statement source file.
            num (int): Number of the statement in the contest starting from 0
        """
        assert(FilePath(directory, 'statement.tex').exists())
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
                if the pdf file cannot be generated.
        """
        if self._pdf.mtime() < self._source.mtime():
            (st, msg) = self.build()
            if not st:
                return None
        return self._pdf

    def __str__(self):
        return self._source.path

    @ui.work('PDF')
    def build(self, blank_page=False):
        """Compile statement latex source
        Args:
           blank_page (Optional[bool]) if true adds a blank page at the end of the
               problem.
        Returns:
           (bool, msg) a tuple containing status code and result message.

        """
        os.environ['OCIMATIC_PROBLEM_NUMBER'] = chr(ord('A')+self._num)
        if blank_page:
            os.environ['OCIMATIC_BLANK_PAGE'] = 'True'
        st = self._compiler(self._source)
        return (st, 'OK' if st else 'FAILED')

    def io_samples(self):
        """Find sample input data in the satement
        Returns:
            List[FilePath]: list of paths
        """
        latex_file = self._source.open('r')
        samples = set()
        for line in latex_file:
            m = re.match(r'[^%]*\\sampleIO{([^}]*)}', line)
            m and samples.add(m.group(1))
        latex_file.close()
        samples = [FilePath(self._directory, s) for s in samples]
        return samples


class LatexCompiler(object):
    """Compiles latex source"""
    def __init__(self, cmd='pdflatex',
                 flags=['--shell-escape', '-interaction=batchmode']):
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
        cmd = 'cd "%s" && %s %s "%s"' % (
            source.directory(), self._cmd, flags, source.name)
        complete = subprocess.run(cmd,
                                  shell=True,
                                  stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        return complete.returncode == 0
