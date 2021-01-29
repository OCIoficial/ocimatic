# coding=UTF-8
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import fnmatch
import tempfile
from typing import Iterable, List, Optional, TypedDict

import ocimatic
from ocimatic import pjson, ui
from ocimatic.checkers import Checker, CppChecker, DiffChecker
from ocimatic.compilers import LatexCompiler
from ocimatic.dataset import Dataset, DatasetPlan, Test
from ocimatic.solutions import CppSolution, Solution


class ContestConfig(TypedDict, total=False):
    phase: str


class Contest:
    """This class represents a contest. A contest is formed by a list of
    tasks and a titlepage. A contest is associated to a directory in the
    filesystem.
    """
    def __init__(self, directory: Path):
        """
        Args:
            directory (Directory): Directory where the contest reside.
        """
        self._directory = directory
        self._config = pjson.load(Path(directory, '.ocimatic_contest'))

        self._init_tasks()

        if 'phase' in self._config:
            os.environ['OCIMATIC_PHASE'] = self._config['phase']

        self._titlepage = Path(directory, 'titlepage.tex')
        self._compiler = LatexCompiler()

    def _init_tasks(self) -> None:
        n = len(self._config['tasks'])
        order = {t: i for i, t in enumerate(self._config['tasks'])}
        dirs = sorted((order.get(dir.name, n), dir) for dir in self._directory.iterdir()
                      if Path(dir, '.ocimatic_task').exists())
        self._tasks = [Task(d, i) for (i, (_, d)) in enumerate(dirs)]

    @staticmethod
    def create_layout(contest_path: Path, config: ContestConfig) -> None:
        """Copies contest skeleton to contest_path and saves specified configurations

        Args:
            contest_path (Filepath)
        """
        ocimatic_dir = Path(__file__).parent
        contest_skel = Path(ocimatic_dir, 'resources', 'contest-skel')
        shutil.copytree(contest_skel,
                        contest_path,
                        ignore=shutil.ignore_patterns('auto'),
                        symlinks=True)
        with Path(contest_path, '.ocimatic_contest').open('w') as config_file:
            json.dump(config, config_file, indent=4)

    def new_task(self, name: str) -> None:
        task_dir = Path(self._directory, name)
        Task.create_layout(task_dir)
        self._config.setdefault('tasks', []).append(name)

    @property
    def tasks(self) -> List['Task']:
        """List[Task]"""
        return self._tasks

    @ui.contest_group('Generating problemset')
    def build_problemset(self) -> None:
        """It builds the titlepage and the statement of all tasks. Then it merges
        all pdfs in a single file.
        """
        self.build_problemset_twoside()
        self.build_problemset_oneside()

    @ui.workgroup('oneside')
    def build_problemset_oneside(self) -> None:
        os.environ['OCIMATIC_SIDENESS'] = 'oneside'
        self.compile_titlepage()

        for task in self._tasks:
            task.build_statement()
        self.merge_pdfs('oneside.pdf')

    @ui.workgroup('twoside')
    def build_problemset_twoside(self) -> None:
        os.environ['OCIMATIC_SIDENESS'] = 'twoside'
        self.compile_titlepage()

        for (i, task) in enumerate(self._tasks):
            last = i == len(self._tasks) - 1
            blank_page = last and ocimatic.config['last_blank_page']
            task.build_statement(blank_page=blank_page)
        self.merge_pdfs('twoside.pdf')

    @ui.contest_group('Building package')
    def package(self) -> bool:
        """Compress statement and dataset of all tasks in a single file"""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            for task in self._tasks:
                task.copy_to(tmpdir)

            self.build_problemset_twoside()
            self.build_problemset_oneside()
            oneside = Path(self._directory, 'oneside.pdf')
            if oneside.exists():
                shutil.copy2(oneside, Path(tmpdir, 'oneside.pdf'))
            twoside = Path(self._directory, 'twoside.pdf')
            if twoside.exists():
                shutil.copy2(twoside, Path(tmpdir, 'twoside.pdf'))

            cmd = 'cd %s && zip -r contest.zip .' % tmpdir
            st = subprocess.call(cmd, stdout=subprocess.DEVNULL, shell=True)
            contest = Path(self._directory, '%s.zip' % self.name)
            shutil.copy2(Path(tmpdir, 'contest.zip'), contest)
        finally:
            shutil.rmtree(tmpdir)

        return st == 0

    @ui.work('PDF', 'titlepage.tex')
    def compile_titlepage(self) -> ui.WorkResult:
        """Compile title page latex
        Returns:
            (bool, msg): Status and result message
        """
        st = self._compiler(self._titlepage)
        return ui.WorkResult(success=st, short_msg='OK' if st else 'FAILED')

    @ui.work('MERGE', '{1}')
    def merge_pdfs(self, filename) -> ui.WorkResult:
        """Merges statements and title page in a single file """
        if not shutil.which('gs'):
            return ui.WorkResult(success=False, short_msg='Cannot find gs')

        pdfs = ' '.join('"%s"' % t.statement.pdf for t in self._tasks if t.statement.pdf)
        titlepage = Path(self._directory, 'titlepage.pdf')
        if titlepage.exists():
            pdfs = '"%s" %s' % (titlepage, pdfs)

        cmd = ('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite'
               ' -dPDFSETTINGS=/prepress -sOutputFile=%s %s') % (Path(self._directory,
                                                                      filename), pdfs)
        complete = subprocess.run(cmd,
                                  shell=True,
                                  timeout=20,
                                  stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  check=False)
        st = complete.returncode == 0
        return ui.WorkResult(success=st, short_msg='OK' if st else 'FAILED')

    @property
    def name(self) -> str:
        """str: Name of the contest"""
        return self._directory.name

    def __str__(self) -> str:
        return self.name

    def find_task(self, name: str) -> Optional['Task']:
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
    def create_layout(task_path: Path) -> None:
        ocimatic_dir = Path(__file__).parent
        skel = Path(ocimatic_dir, 'resources', 'task-skel')
        shutil.copytree(skel, task_path, symlinks=True)

    def __init__(self, directory: Path, num: int):
        """
        Args:
            directory (Directory): Directory where the task resides.
            num (int): Position of the task in the problemset starting from 0.
        """
        self._directory = directory

        self._managers_dir = Path(directory, 'managers')

        self._config = pjson.load(Path(directory, ".ocimatic_task"))

        correct_dir = Path(directory, 'solutions', 'correct')
        self._correct = Solution.get_solutions(self.codename, correct_dir, self._managers_dir)
        partial_dir = Path(directory, 'solutions', 'partial')
        self._partial = Solution.get_solutions(self.codename, partial_dir, self._managers_dir)

        self._checker: Checker = DiffChecker()
        custom_checker = next(self._managers_dir.glob('checker.cpp'), None)
        if custom_checker:
            self._checker = CppChecker(custom_checker)

        self._statement = Statement(Path(directory, 'statement'), num=num, codename=self.codename)

        self._dataset = Dataset(Path(directory, 'dataset'), self._statement.io_samples())

    @property
    def codename(self) -> str:
        return self._directory.name

    @ui.task('Building package')
    def copy_to(self, directory: Path) -> None:
        new_dir = Path(directory, self.codename)
        new_dir.mkdir()

        result = self.compress_dataset()
        if result.success:
            dataset = Path(self._directory, 'dataset', 'data.zip')
            dataset_dst = Path(new_dir, 'data.zip')
            if dataset.exists():
                shutil.copy2(dataset, dataset_dst)

        result = self.build_statement()
        if result.success:
            statement = Path(new_dir, 'statement.pdf')
            shutil.copy2(Path(self._directory, 'statement', 'statement.pdf'), statement)

    @ui.task('Generating dataset input files')
    def gen_input(self) -> None:
        if self._config.get("static_dataset", False):
            ui.fatal_error("Task has a static dataset.")
        testplan = DatasetPlan(Path(self._directory, 'attic'), self._directory,
                               Path(self._directory, 'dataset'))
        testplan.run()

    @ui.task('Validating dataset input files')
    def validate_input(self) -> None:
        testplan = DatasetPlan(Path(self._directory, 'attic'), self._directory,
                               Path(self._directory, 'dataset'))
        testplan.validate_input()

    @ui.work('ZIP')
    def compress_dataset(self, random_sort=False) -> ui.WorkResult:
        """Compress dataset in a single file data.zip"""
        st = self._dataset.compress(random_sort=random_sort)
        return ui.WorkResult(success=st, short_msg='OK' if st else 'FAILED')

    @property
    def name(self) -> str:
        """str: Name of the task"""
        return self._directory.name

    def __str__(self) -> str:
        return self.name

    def solutions(self, partial: bool = False) -> Iterable[Solution]:
        for solution in self._correct:
            yield solution
        if partial:
            for solution in self._partial:
                yield solution

    def get_solution(self, file_path: Path) -> Optional[Solution]:
        return Solution.get_solution(self.codename, file_path, self._managers_dir)

    @property
    def statement(self) -> 'Statement':
        """Statement"""
        return self._statement

    @ui.task('Score Params')
    def score(self) -> None:
        counts = self._dataset.count()
        scores = self._statement.scores()
        if len(scores) != len(counts):
            ui.show_message(
                'Warning',
                "The number of subtasks in the statement doesn't match with the number of " +
                "subtasks in the testplan.", ui.WARNING)
            return

        if len(counts) == len(scores) == 1:
            ui.show_message('Sum', scores[0] / counts[0])
        ui.show_message('GroupMin', [[m, t] for (m, t) in zip(scores, counts)])

    @ui.task('Normalizing')
    def normalize(self) -> None:
        self._dataset.normalize()

    @ui.task('Running solutions')
    def run_solutions(self, solution: Optional[str] = None) -> None:
        """Run all solutions reporting outcome and running time.

        Args:
            solution (Optional[string]): If present it only runs the solutions that
                contain that match this glob pattern.
        """
        for sol in self.solutions(True):
            if not solution or fnmatch.fnmatch(sol.name, solution):
                sol.run(self._dataset, self._checker)

    @ui.task('Checking dataset')
    def check_dataset(self) -> None:
        """Check input/output correctness by running all correct solutions againt all test
        cases and sample input.
        """
        for sol in self.solutions():
            sol.run(self._dataset, self._checker, check=True, sample=True)

    @ui.task('Building solutions')
    def build_solutions(self, pattern=None) -> None:
        """Forces a rebuilding of all solutions, both partial and corrects."""
        for sol in self.solutions(partial=True):
            if pattern is None or fnmatch.fnmatch(sol.name, pattern):
                sol.build()

    @ui.task('Generating expected output')
    def gen_expected(self, sample: bool = False, pattern: str = None):
        """Generate expected outputs files for dataset by running one of the
        correct solutions.
        """
        if self._config.get("static_dataset", False):
            ui.show_message("Skipping", "Task has a static dataset.", ui.WARNING)
            return
        if not self._correct:
            ui.show_message('Skipping', 'No correct solution.', ui.WARNING)
            return
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

    def build_statement(self, blank_page=False) -> ui.WorkResult:
        """Generate pdf for the statement"""
        return self._statement.build(blank_page=blank_page)


class Statement:
    """Represents a statement. A statement is formed by a latex source and a pdf
    file.
    """
    def __init__(self, directory: Path, num: Optional[int] = None, codename: Optional[str] = None):
        """
        Args:
            directory (Directory): Directory to search for statement source file.
            num (int): Number of the statement in the contest starting from 0
        """
        assert Path(directory, 'statement.tex').exists()
        self._source = Path(directory, 'statement.tex')
        self._pdf = self._source.with_suffix('.pdf')
        self._compiler = LatexCompiler()
        self._directory = directory
        self._num = num
        self._codename = codename

    @property
    def pdf(self) -> Optional[Path]:
        """Returns path to pdf file and compiles it if necessary.
        Returns:
            Optional[FilePath]: The file path if the binary is present or None
                if the pdf file cannot be generated.
        """
        if self._pdf.stat().st_mtime < self._source.stat().st_mtime:
            result = self.build()
            if not result.success:
                return None
        return self._pdf

    def __str__(self) -> str:
        return str(self._source)

    @ui.work('PDF')
    def build(self, blank_page: bool = False) -> ui.WorkResult:
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
        return ui.WorkResult(success=st, short_msg='OK' if st else 'FAILED')

    def io_samples(self) -> List[Test]:
        """Find sample input data in the satement
        Returns:
            List[FilePath]: list of paths
        """
        samples = set()
        for line in self._iter_file():
            m = re.match(r'[^%]*\\sampleIO(\[[^\]]*\]){0,2}{([^}]+)}', line)
            if m:
                samples.add(m.group(2))
        return [
            Test(Path(self._directory, f"{s}.in"), Path(self._directory, f"{s}.sol"))
            for s in samples
        ]

    def scores(self) -> List[int]:
        """Finds the scores for the subtasks
        Returns:
            List[int]: List with the scores for each subtask
        """
        scores = []
        for line in self._iter_file():
            m = re.match(r'[^%]*\\subtask{([^}]+)}', line)
            if m:
                scores.append(int(m.group(1)))
        if not scores:
            ui.show_message('Warning', "Couldn't infer the score from the statement, assuming 100.",
                            ui.WARNING)
            scores = [100]

        return scores

    def _iter_file(self) -> Iterable[str]:
        latex_file = self._source.open('r')
        yield from latex_file
        latex_file.close()
