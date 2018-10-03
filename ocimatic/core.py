# coding=UTF-8
import json
import os
import re
import shutil
import subprocess

import ocimatic
from ocimatic import pjson, ui
from ocimatic.checkers import CppChecker, DiffChecker
from ocimatic.compilers import LatexCompiler
from ocimatic.dataset import Dataset, DatasetPlan, SampleData
from ocimatic.filesystem import Directory, FilePath
from ocimatic.solutions import CppSolution, Solution


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
        self._correct = Solution.get_solutions(self.codename, correct_dir, self._managers_dir)
        partial_dir = directory.chdir('solutions/partial')
        self._partial = Solution.get_solutions(self.codename, partial_dir, self._managers_dir)

        self._checker = DiffChecker()
        custom_checker = self._managers_dir.find_file('checker.cpp')
        if custom_checker:
            self._checker = CppChecker(custom_checker)

        self._statement = Statement(directory.chdir('statement'), num=num, codename=self.codename)

        self._dataset = Dataset(
            directory.chdir('dataset', create=True), SampleData(self._statement))

    @property
    def codename(self):
        return self._directory.basename

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
        return Solution.get_solution(self.codename, file_path, self._managers_dir)

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
