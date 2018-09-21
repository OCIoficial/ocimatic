# coding=UTF-8
import json
import os
import re
import shutil
import subprocess
import time as pytime
from contextlib import ExitStack

import ocimatic
from ocimatic import pjson, ui
from ocimatic.filesystem import Directory, FilePath


class Contest:
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
        self._config = pjson.load(FilePath(directory, '.ocimatic_contest'))

        self._init_tasks()

        if 'phase' in self._config:
            os.environ['OCIMATIC_PHASE'] = self._config['phase']

        self._titlepage = FilePath(directory, 'titlepage.tex')
        self._compiler = LatexCompiler()

    def _init_tasks(self):
        keep = []
        dirs = []
        for task in self._config.get('tasks', []):
            path = self._directory.find(task)
            if path and path.isdir():
                keep.append(task)
                dirs.append(path.get_or_create_dir())
        self._config['tasks'] = keep
        self._tasks = [Task(d, i) for (i, d) in enumerate(dirs)]

    @staticmethod
    def create_layout(contest_path, config):
        """Copies contest skeleton to contest_path and saves specified configurations

        Args:
            contest_path (Filepath)
        """
        ocimatic_dir = FilePath(__file__).directory()
        contest_skel = ocimatic_dir.chdir('resources', 'contest-skel')
        contest_skel.copy_tree(contest_path, ['auto'])
        contest_dir = contest_path.get_or_create_dir()
        with FilePath(contest_dir, '.ocimatic_contest').open('w') as config_file:
            json.dump(config, config_file, indent=4)

    def new_task(self, name):
        task_dir = FilePath(self._directory, name)
        Task.create_layout(task_dir)
        self._config.setdefault('tasks', []).append(name)

    @property
    def tasks(self):
        """List[Task]"""
        return self._tasks

    @ui.contest_group('Generating problemset')
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

        for task in self._tasks:
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

    @ui.contest_group('Building package')
    def package(self):
        """Compress statement and dataset of all tasks in a single file"""
        tmpdir = Directory.tmpdir()
        try:
            for task in self._tasks:
                task.copy_to(tmpdir)

            self.build_problemset_twoside()
            self.build_problemset_oneside()
            oneside = FilePath(self._directory, 'oneside.pdf')
            if oneside.exists():
                oneside.copy(FilePath(tmpdir, 'oneside.pdf'))
            twoside = FilePath(self._directory, 'twoside.pdf')
            if twoside.exists():
                twoside.copy(FilePath(tmpdir, 'twoside.pdf'))

            cmd = 'cd %s && zip -r contest.zip .' % tmpdir
            st = subprocess.call(cmd, stdout=subprocess.DEVNULL, shell=True)
            contest = FilePath(self._directory, '%s.zip' % self.name)
            FilePath(tmpdir, 'contest.zip').copy(contest)
        finally:
            tmpdir.rmtree()

        return st == 0

    @ui.work('PDF', 'titlepage.tex')
    def compile_titlepage(self):
        """Compile title page latex
        Returns:
            (bool, msg): Status and result message
        """
        st = self._compiler(self._titlepage)
        return (st, 'OK' if st else 'FAILED')

    @ui.work('MERGE', '{1}')
    def merge_pdfs(self, filename):
        """Merges statements and title page in a single file """
        if not shutil.which('gs'):
            return (False, 'Cannot find gs')

        pdfs = ' '.join('"%s"' % t.statement.pdf for t in self._tasks if t.statement.pdf)
        titlepage = FilePath(self._directory, 'titlepage.pdf')
        if titlepage.exists():
            pdfs = '"%s" %s' % (titlepage, pdfs)

        cmd = ('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite'
               ' -dPDFSETTINGS=/prepress -sOutputFile=%s %s') % (FilePath(
                   self._directory, filename), pdfs)
        complete = subprocess.run(
            cmd,
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


class Task:
    """This class represents a task. A task consists of a statement,
    a list of correct and partial solutions and a dataset. A task is
    associated to a directory in the filesystem.
    """

    @staticmethod
    def create_layout(task_path):
        ocimatic_dir = FilePath(__file__).directory()
        skel = ocimatic_dir.chdir('resources', 'task-skel')
        skel.copy_tree(task_path)

    def __init__(self, directory, num):
        """
        Args:
            directory (Directory): Directory where the task resides.
            num (int): Position of the task in the problemset starting from 0.
        """
        self._directory = directory

        self._managers_dir = directory.chdir('managers')

        correct_dir = directory.chdir('solutions/correct')
        self._correct = Solution.get_solutions(correct_dir, self._managers_dir)
        partial_dir = directory.chdir('solutions/partial')
        self._partial = Solution.get_solutions(partial_dir, self._managers_dir)

        self._checker = DiffChecker()
        custom_checker = self._managers_dir.find_file('checker.cpp')
        if custom_checker:
            self._checker = CppChecker(custom_checker)

        self._statement = Statement(
            directory.chdir('statement'), num=num, codename=self._directory.basename)

        self._dataset = Dataset(
            directory.chdir('dataset', create=True), SampleData(self._statement))

    @ui.task('Building package')
    def copy_to(self, directory):
        new_dir = directory.mkdir(str(self))

        (st, _) = self.compress_dataset()
        if st:
            dataset = FilePath(self._directory.chdir('dataset'), 'data.zip')
            dataset_dst = FilePath(new_dir, 'data.zip')
            if dataset.exists():
                dataset.copy(dataset_dst)

        (st, _) = self.build_statement()
        if st:
            statement = FilePath(new_dir, 'statement.pdf')
            FilePath(self._directory.chdir('statement'), 'statement.pdf').copy(statement)

    @ui.task('Generating dataset input files')
    def gen_input(self):
        testplan = DatasetPlan(
            self._directory.chdir('attic'), self._directory, self._directory.chdir('dataset'))
        testplan.run()

    @ui.task('Validating dataset input files')
    def validate_input(self):
        testplan = DatasetPlan(
            self._directory.chdir('attic'), self._directory, self._directory.chdir('dataset'))
        testplan.validate_input()

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

    def get_solution(self, file_path):
        return Solution.get_solution(file_path, self._managers_dir)

    @property
    def statement(self):
        """Statement"""
        return self._statement

    @ui.task('Counting')
    def count(self):
        self._dataset.count()

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
        for sol in self.solutions(partial):
            if not pattern or pattern.lower() in sol.name.lower():
                self.run_solution(sol)

    def run_solution(self, solution):
        solution.run(self._dataset, self._checker)

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
            if pattern is None or pattern.lower() in sol.name.lower():
                sol.build()

    @ui.task('Generating expected output')
    def gen_expected(self, sample=False, pattern=None):
        """Generate expected outputs files for dataset by running one of the
        correct solutions.
        """
        if not self._correct:
            ui.fatal_error('No correct solution.')
        generator = None
        if pattern:
            for sol in self._correct:
                if pattern.lower() in sol.name.lower():
                    generator = sol
                    break
        else:
            cpp = [sol for sol in self._correct if isinstance(sol, CppSolution)]
            if cpp:
                generator = cpp[0]
        if not generator:
            generator = self._correct[0]
        generator.gen_expected(self._dataset, sample=sample)

    def build_statement(self, blank_page=False):
        """Generate pdf for the statement"""
        return self._statement.build(blank_page=blank_page)


class Solution:
    """Abstract class to represent a solution
    """

    def __init__(self, source):
        self._source = source

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
        return [
            solution for file_path in solutions_dir.lsfile()
            for solution in [Solution.get_solution(file_path, managers_dir)] if solution
        ]

    @staticmethod
    def get_solution(file_path, managers_dir):
        if file_path.ext == CppSolution.ext:
            return CppSolution(file_path, managers_dir)
        if file_path.ext == JavaSolution.ext:
            return JavaSolution(file_path, managers_dir)
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
        raise NotImplementedError(
            "Class %s doesn't implement get_runnable()" % (self.__class__.__name__))

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
        raise NotImplementedError(
            "Class %s doesn't implement get_runnable()" % (self.__class__.__name__))

    def build_time(self):
        raise NotImplementedError(
            "Class %s doesn't implement build_time()" % (self.__class__.__name__))

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


class CppCompiler:
    """Compiles C++ code
    """

    def __init__(self, flags=('-std=c++11', '-O2')):
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

        complete = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
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
        # TODO: implement managers for java
        del managers
        super().__init__(source)
        assert source.ext == self.ext
        self._source = source
        self._compiler = JavaCompiler()
        # self._grader = managers.find_file('grader.cpp')
        self._classname = self._source.rootname()
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


class JavaCompiler:
    """Compiles Java code
    """

    def __init__(self, flags=()):
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

        complete = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        return complete.returncode == 0


class Dataset:
    """Test data"""

    def __init__(self, directory, sampledata=None, in_ext='.in', sol_ext='.sol'):
        """
        Args:
            directory (Directory): dataset directory.
            sampledata (Optional[SampleData]): optional sampledata
        """
        self._directory = directory
        self._in_ext = in_ext
        self._sol_ext = sol_ext
        self._subtasks = [Subtask(d, in_ext, sol_ext) for d in directory.lsdir()]
        self._sampledata = sampledata

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

    def mtime(self):
        mtime = -1
        for subtask in self._subtasks:
            mtime = max(mtime, subtask.mtime())
        return mtime

    def compress(self, in_ext=None, sol_ext=None):
        """Compress all test cases in this dataset in a single zip file.
        The basename of the corresponding subtask subdirectory is prepended
        to each file.
        """
        in_ext = in_ext or self._in_ext
        sol_ext = sol_ext or self._sol_ext
        dst_file = FilePath(self._directory, 'data.zip')
        if dst_file.exists() and dst_file.mtime() >= self.mtime():
            return True

        tmpdir = Directory.tmpdir()
        try:
            copied = 0
            for subtask in self._subtasks:
                copied += subtask.copy_to(tmpdir)

            if not copied:
                # ui.show_message("Warning", "no files in dataset", ui.WARNING)
                return True

            cmd = 'cd %s && zip data.zip *%s *%s' % (tmpdir, in_ext, sol_ext)
            st = subprocess.call(cmd, stdout=subprocess.DEVNULL, shell=True)
            FilePath(tmpdir, 'data.zip').copy(dst_file)
        finally:
            tmpdir.rmtree()

        return st == 0

    def count(self):
        for st in self._subtasks:
            st.count()

    def normalize(self):
        for subtask in self._subtasks:
            subtask.normalize()
        self._sampledata.normalize()


class Subtask:
    def __init__(self, directory, in_ext='.in', sol_ext='.sol'):
        self._tests = []
        self._in_ext = in_ext
        self._sol_ext = sol_ext
        for f in directory.lsfile('*' + self._in_ext):
            self._tests.append(Test(f, f.chext(sol_ext)))
        self._name = directory.basename

    def copy_to(self, directory):
        copied = 0
        for test in self._tests:
            if test.expected_path.exists():
                in_name = "%s-%s" % (self._name, test.in_path.name)
                sol_name = "%s-%s" % (self._name, test.expected_path.name)
                test.in_path.copy(FilePath(directory, in_name))
                test.expected_path.copy(FilePath(directory, sol_name))
                copied += 1
        return copied

    def mtime(self):
        mtime = -1
        for test in self._tests:
            mtime = max(mtime, test.mtime())
        return mtime

    def normalize(self):
        for test in self._tests:
            test.normalize()

    @ui.work('Gen')
    def count(self):
        return (True, len(self._tests))

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
    # FIXME: this shouldn't inherit directly from Subtask as the initializer is completely different.
    # Maybe both should a have a common parent.
    def __init__(self, statement, in_ext='.in', sol_ext='.sol'):  # pylint: disable=super-init-not-called
        self._in_ext = in_ext
        self._sol_ext = sol_ext
        tests = statement.io_samples() if statement else []
        self._tests = [Test(f.chext(in_ext), f.chext(sol_ext)) for f in tests]

    def __str__(self):
        return 'Sample'


class Test:
    """A single test file. Expected output file may not exist"""

    def __init__(self, in_path, expected_path):
        """
        Args:
            in_path (FilePath)
            expected_path (FilePath)
        """
        assert in_path.exists()
        self._in_path = in_path
        self._expected_path = expected_path

    def __str__(self):
        return str(self._in_path)

    def mtime(self):
        if self._expected_path.exists():
            return max(self._in_path.mtime(), self._expected_path.mtime())
        return self._in_path.mtime()

    @property
    def directory(self):
        """Directory: directory where this test reside"""
        return self._in_path.directory()

    @ui.work('Gen')
    def gen_expected(self, runnable):
        """Run binary with this test as input to generate expected output file
        Args:
            runnable (Runnable)
        Returns:
            (bool, msg): A tuple containing status and result message.
        """
        (st, _, errmsg) = runnable.run(
            self.in_path, self.expected_path, timeout=ocimatic.config['timeout'])
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
        if not self.expected_path.exists():
            out_path.remove()
            return (False, 'No expected output file')

        (st, time, errmsg) = runnable.run(
            self.in_path, out_path, timeout=ocimatic.config['timeout'])

        # Execution failed
        if not st:
            if check:
                return (st, errmsg)
            return (st, '%s' % errmsg)

        (st, outcome, checkmsg) = checker(self.in_path, self.expected_path, out_path)
        # Checker failed
        if not st:
            msg = 'Failed to run checker: %s' % checkmsg
            return (st, msg)

        st = outcome == 1.0
        if check:
            msg = 'OK' if st else 'FAILED'
            return (st, msg)

        msg = '%s [%.2fs]' % (outcome, time)
        if checkmsg:
            msg += ' - %s' % checkmsg
        return (st, msg)

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
        tounix_input = 'dos2unix "%s"' % self.in_path
        tounix_expected = 'dos2unix "%s"' % self.expected_path
        sed_input = "sed -i -e '$a\\' \"%s\"" % self.in_path
        sed_expected = "sed -i -e '$a\\' \"%s\"" % self.expected_path
        null = subprocess.DEVNULL
        st = subprocess.call(tounix_input, stdout=null, stderr=null, shell=True)
        st += subprocess.call(sed_input, stdout=null, stderr=null, shell=True)
        if self.expected_path.exists():
            st += subprocess.call(tounix_expected, stdout=null, stderr=null, shell=True)
            st += subprocess.call(sed_expected, stdout=null, stderr=null, shell=True)
        return (st == 0, 'OK' if st == 0 else 'FAILED')


# FIXME: Refactor class. This should allow to re-enable some pylint checks
class DatasetPlan:
    """Functionality to read and run a plan for generating dataset."""

    def __init__(self, directory, task_directory, dataset_directory, filename='testplan.txt'):
        self._directory = directory
        self._testplan_path = FilePath(directory, filename)
        if not self._testplan_path.exists():
            ui.fatal_error('No such file plan for creating dataset: "%s"' % self._testplan_path)
        self._task_directory = task_directory
        self._dataset_directory = dataset_directory
        self._cpp_compiler = CppCompiler()
        self._java_compiler = JavaCompiler()

    def test_filepath(self, stn, group, i):
        st_dir = FilePath(self._dataset_directory, 'st%d' % stn).get_or_create_dir()
        return FilePath(st_dir, '%s-%d.in' % (group, i))

    def validate_input(self):
        (_, cmds) = self.parse_file()
        for (st, subtask) in sorted(cmds.items()):
            self.validate_subtask(st, subtask)

    @ui.workgroup('Subtask {1}')
    def validate_subtask(self, stn, subtask):
        validator = None
        if subtask['validator']:
            (validator, msg) = self.build_validator(subtask['validator'])
            if validator is None:
                ui.show_message('Warning', 'Failed to build validator: %s' % msg, ui.WARNING)
        else:
            ui.show_message('Info', 'No validator specified', ui.INFO)
        if validator:
            for (group, tests) in sorted(subtask['groups'].items()):
                for (i, _) in enumerate(tests, 1):
                    test_file = self.test_filepath(stn, group, i)
                    self.validate_test_input(test_file, validator)

    @ui.work('Validating', '{1}')
    def validate_test_input(self, test_file, validator):
        if not test_file.exists():
            return False, 'Test file does not exist'
        (st, _time, msg) = validator.run(test_file, None)
        return st, msg

    def build_validator(self, source):
        fp = FilePath(self._directory, source)
        if not fp.exists():
            return (None, 'File does not exists.')
        if fp.ext == '.cpp':
            binary = fp.chext('.bin')
            if binary.mtime() < fp.mtime() and not self._cpp_compiler(fp, binary):
                return (None, 'Failed to build validator.')
            return (Runnable(binary), 'OK')
        if fp.ext in ['.py', '.py3']:
            return (Runnable('python3', [str(source)]), 'OK')
        if fp.ext == '.py2':
            return (Runnable('python2', [str(source)]), 'OK')
        return (None, 'Not supported source file.')

    def run(self):
        (subtasks, cmds) = self.parse_file()

        for stn in range(1, subtasks + 1):
            dire = FilePath(self._dataset_directory, 'st%d' % stn).get_or_create_dir()
            dire.clear()

        if not cmds:
            ui.show_message("Warning", 'no commands were executed for the plan.', ui.WARNING)

        for (stn, subtask) in sorted(cmds.items()):
            self.run_subtask(stn, subtask)

    @ui.workgroup('Subtask {1}')
    def run_subtask(self, stn, subtask):
        groups = subtask['groups']
        for (group, tests) in sorted(groups.items()):
            for (i, test) in enumerate(tests, 1):
                cmd = test['cmd']
                test_file = self.test_filepath(stn, group, i)
                if cmd == 'copy':
                    self.copy(test['file'], test_file)
                elif cmd == 'echo':
                    self.echo(test['args'], test_file)
                else:
                    test['args'].insert(0, '%s-%s-%s' % (stn, group, i))
                    source = FilePath(self._directory, test['source'])
                    if cmd == 'cpp':
                        self.run_cpp_generator(source, test['args'], test_file)
                    elif cmd in ['py', 'py2', 'py3']:
                        self.run_py_generator(source, test['args'], test_file, cmd)
                    elif cmd == 'java':
                        self.run_java_generator(source, test['args'], test_file)
                    elif cmd == 'run':
                        bin_path = FilePath(self._directory, test['bin'])
                        self.run_bin_generator(bin_path, test['args'], test_file)
                    else:
                        ui.fatal_error('unexpected command when running plan: %s ' % cmd)

    @ui.work('Copy', '{1}')
    def copy(self, src, dst):
        fp = FilePath(self._task_directory, src)
        if not fp.exists():
            return (False, 'No such file')
        try:
            fp.copy(dst)
            (st, msg) = (True, 'OK')
            return (st, msg)
        except Exception:  # pylint: disable=broad-except
            return (False, 'Error when copying file')

    @ui.work('Echo', '{1}')
    def echo(self, args, dst):
        with dst.open('w') as test_file:
            test_file.write(' '.join(args) + '\n')
            (st, msg) = (True, 'Ok')
            return (st, msg)

    @ui.work('Gen', '{1}')
    def run_cpp_generator(self, source, args, dst):
        if not source.exists():
            return (False, 'No such file')
        binary = source.chext('.bin')
        if binary.mtime() < source.mtime():
            st = self._cpp_compiler(source, binary)
            if not st:
                return (st, 'Failed to build generator')

        (st, _time, msg) = Runnable(binary).run(None, dst, args)
        return (st, msg)

    @ui.work('Gen', '{1}')
    def run_py_generator(self, source, args, dst, cmd):
        if not source.exists():
            return (False, 'No such file')
        python = 'python2' if cmd == 'py2' else 'python3'
        (st, _time, msg) = Runnable(python, [str(source)]).run(None, dst, args)
        return (st, msg)

    @ui.work('Gen', '{1}')
    def run_java_generator(self, source, args, dst):
        if not source.exists():
            return (False, 'No such file')
        bytecode = source.chext('.class')
        if bytecode.mtime() < source.mtime():
            st = self._java_compiler(source)
            if not st:
                return (st, 'Failed to build generator')

        classname = bytecode.rootname()
        classpath = str(bytecode.directory().path())
        (st, _time, msg) = Runnable('java', ['-cp', classpath, classname]).run(None, dst, args)
        return (st, msg)

    @ui.work('Gen', '{1}')
    def run_bin_generator(self, bin_path, args, dst):
        if not bin_path.exists():
            return (False, 'No such file')
        if not Runnable.is_callable(bin_path):
            return (False, 'Cannot run file, it may not have correct permissions')
        (st, _time, msg) = Runnable(bin_path).run(None, dst, args)
        return (st, msg)

    def parse_file(self):  # pylint: disable=too-many-locals,too-many-branches
        """
        Args:
            path (FilePath)
        """
        cmds = {}
        st = 0
        for (lineno, line) in enumerate(self._testplan_path.open('r').readlines(), 1):
            line = line.strip()
            subtask_header = re.compile(r'\s*\[\s*Subtask\s*(\d+)\s*(?:-\s*([^\]\s]+))?\s*\]\s*')
            cmd_line = re.compile(r'\s*([^;\s]+)\s*;\s*(\S+)(:?\s+(.*))?')
            comment = re.compile(r'\s*#.*')

            if not line:
                continue
            if not comment.fullmatch(line):
                header_match = subtask_header.fullmatch(line)
                cmd_match = cmd_line.fullmatch(line)
                if header_match:
                    found_st = int(header_match.group(1))
                    validator = header_match.group(2)
                    if st + 1 != found_st:
                        ui.fatal_error('line %d: found subtask %d, but subtask %d was expected' %
                                       (lineno, found_st, st + 1))
                    st += 1
                    cmds[st] = {'validator': validator, 'groups': {}}
                elif cmd_match:
                    if st == 0:
                        ui.fatal_error(
                            'line %d: found command before declaring a subtask.' % lineno)
                    group = cmd_match.group(1)
                    cmd = cmd_match.group(2)
                    args = (cmd_match.group(3) or '').split()
                    if group not in cmds[st]['groups']:
                        cmds[st]['groups'][group] = []

                    if cmd == 'copy':
                        if len(args) > 2:
                            ui.fatal_error(
                                'line %d: command copy expects exactly one argument.' % lineno)
                        cmds[st]['groups'][group].append({
                            'cmd': 'copy',
                            'file': args[0],
                        })
                    elif cmd == 'echo':
                        cmds[st]['groups'][group].append({'cmd': 'echo', 'args': args})
                    else:
                        f = FilePath(self._directory, cmd)
                        if f.ext in ['.cpp', '.java', '.py', '.py2', '.py3']:
                            cmds[st]['groups'][group].append({
                                'cmd': f.ext[1:],
                                'source': cmd,
                                'args': args
                            })
                        else:
                            cmds[st]['groups'][group].append({
                                'cmd': 'run',
                                'bin': cmd,
                                'args': args
                            })
                else:
                    ui.fatal_error('line %d: error while parsing line `%s`\n' % (lineno, line))
        return (st, cmds)


class Checker:
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
        NotImplementedError("Class %s doesn't implement __call__()" % (self.__class__.__name__))


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
        assert shutil.which('diff')
        assert in_path.exists()
        assert expected_path.exists()
        assert out_path.exists()
        complete = subprocess.run(
            ['diff', str(expected_path), str(out_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        # @TODO(NL 20/10/1990) Check if returncode is nonzero because there
        # is a difference between files or because there was an error executing
        # diff.
        st = complete.returncode == 0
        outcome = 1.0 if st else 0.0
        return (True, outcome, '')


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
        assert in_path.exists()
        assert expected_path.exists()
        assert out_path.exists()
        if self._binary_path.mtime() < self._source.mtime():
            if not self.build():
                return (False, 0.0, "Failed to build checker")
        complete = subprocess.run(
            [str(self._binary_path),
             str(in_path), str(expected_path),
             str(out_path)],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        ret = complete.returncode
        st = ret == 0
        if st:
            outcome = float(complete.stdout)
            msg = complete.stderr
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

        return (st, outcome, msg)

    def build(self):
        """Build source of the checker
        Returns:
            bool: True if compilation is successful. False otherwise
        """
        return self._compiler(self._source, self._binary_path)


SIGNALS = {
    1: 'SIGHUP',
    2: 'SIGINT',
    3: 'SIGQUIT',
    4: 'SIGILL',
    5: 'SIGTRAP',
    6: 'SIGABRT',
    7: 'SIGEMT',
    8: 'SIGFPE',
    9: 'SIGKILL',
    10: 'SIGBUS',
    11: 'SIGSEGV',
    12: 'SIGSYS',
    13: 'SIGPIPE',
    14: 'SIGALRM',
    15: 'SIGTERM',
    16: 'SIGURG',
    17: 'SIGSTOP',
    18: 'SIGTSTP',
    19: 'SIGCONT',
    20: 'SIGCHLD',
    21: 'SIGTTIN',
    22: 'SIGTTOU',
    23: 'SIGIO',
    24: 'SIGXCPU',
    25: 'SIGXFSZ',
    26: 'SIGVTALRM',
    27: 'SIGPROF',
    28: 'SIGWINCH',
    29: 'SIGINFO',
    30: 'SIGUSR1',
    31: 'SIGUSR2',
}


class Runnable:
    """An entity that may be executed redirecting stdin and stdout to specific
    files.
    """

    @staticmethod
    def is_callable(file_path):
        return shutil.which(str(file_path)) is not None

    def __init__(self, command, args=None):
        """
        Args:
            bin_path (FilePath|string): Command to execute.
            args (List[string]): List of arguments to pass to the command.
        """
        args = args or []
        command = str(command)
        assert shutil.which(command)
        self._cmd = [command] + args

    def __str__(self):
        return self._cmd[0]

    def run(self, in_path, out_path, args=None, timeout=None):  # pylint: disable=too-many-locals
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
        args = args or []
        assert in_path is None or in_path.exists()
        with ExitStack() as stack:
            if in_path is None:
                in_path = FilePath('/dev/null')
            in_file = stack.enter_context(in_path.open('r'))
            if not out_path:
                out_path = FilePath('/dev/null')
            out_file = stack.enter_context(out_path.open('w'))

            start = pytime.monotonic()
            self._cmd.extend(args)
            try:
                complete = subprocess.run(
                    self._cmd,
                    timeout=timeout,
                    stdin=in_file,
                    stdout=out_file,
                    universal_newlines=True,
                    stderr=subprocess.PIPE)
            except subprocess.TimeoutExpired:
                return (False, pytime.monotonic() - start, 'Execution timed out')
            time = pytime.monotonic() - start
            ret = complete.returncode
            status = ret == 0
            msg = 'OK'
            if not status:
                stderr = complete.stderr.strip('\n')
                if stderr and len(stderr) < 100:
                    msg = stderr
                else:
                    if ret < 0:
                        sig = -ret
                        msg = 'Execution killed with signal %d' % sig
                        if sig in SIGNALS:
                            msg += ': %s' % SIGNALS[sig]
                    else:
                        msg = 'Execution ended with error (return code %d)' % ret
            return (status, time, msg)


class Statement:
    """Represents a statement. A statement is formed by a latex source and a pdf
    file.
    """

    def __init__(self, directory, num=None, codename=None):
        """
        Args:
            directory (Directory): Directory to search for statement source file.
            num (int): Number of the statement in the contest starting from 0
        """
        assert FilePath(directory, 'statement.tex').exists()
        self._source = FilePath(directory, 'statement.tex')
        self._pdf = self._source.chext('.pdf')
        self._compiler = LatexCompiler()
        self._directory = directory
        self._num = num
        self._codename = codename

    @property
    def pdf(self):
        """Returns path to pdf file and compiles it if necessary.
        Returns:
            Optional[FilePath]: The file path if the binary is present or None
                if the pdf file cannot be generated.
        """
        if self._pdf.mtime() < self._source.mtime():
            (st, _msg) = self.build()
            if not st:
                return None
        return self._pdf

    def __str__(self):
        return str(self._source)

    @ui.work('PDF')
    def build(self, blank_page=False):
        """Compile statement latex source
        Args:
           blank_page (Optional[bool]) if true adds a blank page at the end of the
               problem.
        Returns:
           (bool, msg) a tuple containing status code and result message.

        """
        if self._num is not None:
            os.environ['OCIMATIC_PROBLEM_NUMBER'] = chr(ord('A') + self._num)
        if self._codename:
            os.environ['OCIMATIC_CODENAME'] = self._codename
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
            m = re.match(r'[^%]*\\sampleIO(\[[^\]]*\]){0,2}{([^}]+)}', line)
            if m:
                samples.add(m.group(2))
        latex_file.close()
        return [FilePath(self._directory, s) for s in samples]


class LatexCompiler:
    """Compiles latex source"""

    def __init__(self, cmd='pdflatex', flags=('--shell-escape', '-interaction=batchmode')):
        """
        Args:
            cmd (str): command to compile files. default to pdflatex
            flags (List[str]): list of flags to pass to command
        """
        self._cmd = cmd
        self._flags = flags

    def __call__(self, source):
        """It compiles a latex source leaving the pdf in the same directory of
        the source.
        Args:
            source (FilePath): path of file to compile
        """
        flags = ' '.join(self._flags)
        cmd = 'cd "%s" && %s %s "%s"' % (source.directory(), self._cmd, flags, source.name)
        complete = subprocess.run(
            cmd,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        return complete.returncode == 0
