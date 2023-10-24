# coding=UTF-8
import json
import os
import re
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict

import pypdf

import ocimatic
from ocimatic import pjson, ui
from ocimatic.checkers import Checker
from ocimatic.dataset import Dataset, Test
from ocimatic.solutions import Solution
from ocimatic.source_code import CppSource, JavaSource, LatexSource, RustSource
from ocimatic.testplan import Testplan


class ContestConfig(TypedDict, total=False):
    phase: str


def change_directory() -> Tuple[Path, Optional[Path]]:
    """Changes directory to the contest root and returns the absolute path of the
    last directory before reaching the root, this correspond to the directory
    of the problem in which ocimatic was called. If the function reach system
    root the program exists with an error.
    """
    last_dir = None
    curr_dir = Path.cwd()
    while not Path(curr_dir, '.ocimatic_contest').exists():
        last_dir = curr_dir
        curr_dir = curr_dir.parent
        if curr_dir.samefile(last_dir):
            ui.fatal_error('ocimatic was not called inside a contest.')
    ocimatic.config['contest_root'] = curr_dir
    return (curr_dir, last_dir)


class Contest:
    """This class represents a contest. A contest is formed by a list of
    tasks and a titlepage. A contest is associated to a directory in the
    filesystem.
    """

    def __init__(self, directory: Path):
        self._directory = directory
        self._config = pjson.load(Path(directory, '.ocimatic_contest'))

        self._init_tasks()

        if 'phase' in self._config:
            os.environ['OCIMATIC_PHASE'] = self._config['phase']

        self._titlepage = LatexSource(Path(directory, 'titlepage.tex'))

    def _init_tasks(self) -> None:
        tasks = self._config.get('tasks', [])
        n = len(tasks)
        order = {t: i for i, t in enumerate(tasks)}
        dirs = sorted((order.get(dir.name, n), dir) for dir in self._directory.iterdir()
                      if Path(dir, '.ocimatic_task').exists())
        self._tasks = [Task(d, i) for (i, (_, d)) in enumerate(dirs)]

    @property
    def directory(self) -> Path:
        return self._directory

    @staticmethod
    def create_layout(dest: Path, config: ContestConfig) -> None:
        """Copy contest skeleton to `dest` and save configuration"""
        ocimatic_dir = Path(__file__).parent
        contest_skel = Path(ocimatic_dir, 'resources', 'contest-skel')
        shutil.copytree(contest_skel, dest, ignore=shutil.ignore_patterns('auto'), symlinks=True)
        with Path(dest, '.ocimatic_contest').open('w') as config_file:
            json.dump(config, config_file, indent=4)

    def new_task(self, name: str) -> None:
        task_dir = Path(self._directory, name)
        Task.create_layout(task_dir)
        self._config.setdefault('tasks', []).append(name)

    @property
    def tasks(self) -> List['Task']:
        return self._tasks

    @ui.contest_group('Generating problemset')
    def build_problemset(self) -> None:
        """Build titlepage and statement of all tasks. Then merge
        all pdfs into a single pdf.
        """
        self.build_problemset_twoside()
        self.build_problemset_oneside()

    @ui.workgroup('oneside')
    def build_problemset_oneside(self) -> None:
        os.environ['OCIMATIC_SIDENESS'] = 'oneside'
        self.compile_titlepage()

        for task in self._tasks:
            task.statement.build(blank_page=False)
        self.merge_pdfs('oneside.pdf')

    @ui.workgroup('twoside')
    def build_problemset_twoside(self) -> None:
        os.environ['OCIMATIC_SIDENESS'] = 'twoside'
        self.compile_titlepage()

        for (i, task) in enumerate(self._tasks):
            last = i == len(self._tasks) - 1
            blank_page = last and ocimatic.config['last_blank_page']
            task.statement.build(blank_page=blank_page)
        self.merge_pdfs('twoside.pdf')

    @ui.contest_group('Building package')
    def package(self) -> None:
        """Package statements and datasets of all tasks into a single zip file"""
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

            shutil.make_archive(self.name, "zip", tmpdir)
        finally:
            shutil.rmtree(tmpdir)

    @ui.work('LATEX', 'titlepage.tex')
    def compile_titlepage(self) -> ui.WorkResult:
        """Compile title page latex"""
        success = self._titlepage.compile() is not None
        return ui.WorkResult(success=success, short_msg='OK' if success else 'FAILED')

    @ui.work('MERGE', '{1}')
    def merge_pdfs(self, filename: str) -> ui.WorkResult:
        """Merge titlepage and statements pdfs into a single file """
        try:
            merger = pypdf.PdfFileMerger()
            for task in self._tasks:
                merger.append(task.statement.pdf)
            titlepage = Path(self._directory, 'titlepage.pdf')
            if titlepage.exists():
                merger.append(titlepage)

            merger.write(Path(self._directory, filename))
            merger.close()
            return ui.WorkResult(success=True, short_msg='OK')
        except Exception as exc:
            return ui.WorkResult(success=False, short_msg="FAILED", long_msg=str(exc))

    @property
    def name(self) -> str:
        """Name of the contest"""
        return self._directory.name

    def __str__(self) -> str:
        return self.name

    def find_task(self, name: str) -> Optional['Task']:
        """find task with given name."""
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
        self._directory = directory

        self._managers_dir = Path(directory, 'managers')

        self._config = pjson.load(Path(directory, ".ocimatic_task"))

        correct_dir = Path(directory, 'solutions', 'correct')
        self._correct = Solution.load_solutions_in_dir(self.codename, correct_dir,
                                                       self._managers_dir)
        partial_dir = Path(directory, 'solutions', 'partial')
        self._partial = Solution.load_solutions_in_dir(self.codename, partial_dir,
                                                       self._managers_dir)

        self._checker = Checker.find_in_directory(self._managers_dir)

        self._statement = Statement(Path(directory, 'statement'), num=num, codename=self.codename)

        self._dataset = Dataset(Path(directory, 'dataset'), self._statement.io_samples())

    @property
    def codename(self) -> str:
        return self._directory.name

    @ui.task('Building package')
    def copy_to(self, directory: Path) -> None:
        new_dir = Path(directory, self.codename)
        new_dir.mkdir()

        result = self._dataset.compress(False)
        if result.success:
            dataset = Path(self._directory, 'dataset', 'data.zip')
            dataset_dst = Path(new_dir, 'data.zip')
            if dataset.exists():
                shutil.copy2(dataset, dataset_dst)

        result = self.statement.build(blank_page=False)
        if result.success:
            statement = Path(new_dir, 'statement.pdf')
            shutil.copy2(Path(self._directory, 'statement', 'statement.pdf'), statement)

    @ui.task('Generating dataset input files')
    def run_testplan(self, subtask: Optional[int]) -> None:
        if self._config.get("static_dataset", False):
            ui.fatal_error("Task has a static dataset.")
        testplan = Testplan(Path(self._directory, 'attic'), self._directory,
                            Path(self._directory, 'dataset'))
        testplan.run(subtask)

    def load_solution_for_path(self, path: Path) -> Optional[Solution]:
        """Search for a solution matching a path. The behavior depends on whether the path
        is absolute or relative. If absolute, it will match a solution for the
        corresponding path. If it is relative, it will try to match the path
        relative to the following locations, in order, until it finds a match or it fails to
        find one:
        1. <task>/solutions/correct
        2. <task>/solutions/partial
        3. <task>/solutions/
        4. <task>
        4. <cwd>
        Where <task> is the path of the current task and <cwd> is the current working
        directory.
        """
        if path.is_absolute():
            return Solution.load(self.codename, path, self._managers_dir)

        for dir in [
                Path(self._directory, 'solutions', 'correct'),
                Path(self._directory, 'solutions', 'partial'),
                Path(self._directory, 'solutions'), self._directory,
                Path.cwd()
        ]:
            sol = Solution.load(self.codename, Path(dir, path), self._managers_dir)
            if sol:
                return sol
        return None

    @ui.task('Validating dataset input files')
    def validate_input(self, subtask: Optional[int]) -> None:
        testplan = Testplan(Path(self._directory, 'attic'), self._directory,
                            Path(self._directory, 'dataset'))
        self._dataset.validate(testplan.validators(), subtask)
        # testplan.validate_input(subtask)

    @ui.task('Compressing dataset')
    def compress_dataset(self, random_sort: bool) -> None:
        """Compress dataset into a single file"""
        self._dataset.compress(random_sort=random_sort)

    @property
    def name(self) -> str:
        """Name of the task"""
        return self._directory.name

    def __str__(self) -> str:
        return self.name

    def solutions(self, partial: bool = False) -> Iterable[Solution]:
        for solution in self._correct:
            yield solution
        if partial:
            for solution in self._partial:
                yield solution

    @property
    def statement(self) -> 'Statement':
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
            ui.show_message('Sum', str(scores[0] / counts[0]))
        ui.show_message('GroupMin', str([[m, t] for (m, t) in zip(scores, counts)]))

    @ui.task('Normalizing')
    def normalize(self) -> None:
        self._dataset.normalize()

    @ui.task('Running solution')
    def run_solution(self, solution: Path) -> None:
        """Run all solutions reporting outcome and running time."""
        sol = self.load_solution_for_path(solution)
        if not sol:
            return ui.show_message('Error', 'Solution not found', ui.ERROR)

        results = sol.run(self._dataset, self._checker)
        if results:
            stats = results.runtime_stats()
            ui.show_message('Max running time', f" {stats.max:.3f}s")
            ui.show_message('Min running time', f" {stats.min:.3f}s")
            ui.show_message('Mean running time', f"{stats.mean:.3f}s")

    @ui.task('Checking dataset')
    def check_dataset(self) -> None:
        """Check input/output correctness by running all correct solutions againt all test
        cases and sample input.
        """
        for sol in self.solutions():
            sol.run(self._dataset, self._checker, check_mode=True, run_on_sample_data=True)

    @ui.task('Building solutions')
    def build_solution(self, solution: Path) -> None:
        """Force compilation of solutions."""
        sol = self.load_solution_for_path(solution)
        if sol:
            sol.build()

    @ui.task('Generating expected output')
    def gen_expected(self, sample: bool = False, solution: Optional[Path] = None) -> None:
        """Generate expected outputs files for the dataset by running one of the
        correct solutions.
        """
        if self._config.get("static_dataset", False):
            ui.show_message("Skipping", "Task has a static dataset.", ui.WARNING)
            return
        if not self._correct:
            ui.show_message('Skipping', 'No correct solution.', ui.WARNING)
            return
        generator = None
        if solution:
            generator = self.load_solution_for_path(solution)
        else:
            keys: Dict[type, int] = {CppSource: 0, RustSource: 1, JavaSource: 0}
            sols = sorted(self._correct, key=lambda sol: keys.get(type(sol.source), len(keys)))
            generator = sols[0] if sols else None

        if not generator:
            ui.fatal_error("solution not found")
        generator.gen_expected(self._dataset, sample=sample)

    @ui.task('Building statement')
    def build_statement(self, blank_page: bool = False) -> None:
        """Generate pdf for the statement"""
        self._statement.build(blank_page=blank_page)


class Statement:
    """Represents a statement. A statement is composed by a latex source and a pdf
    file.
    """

    def __init__(self, directory: Path, num: Optional[int] = None, codename: Optional[str] = None):
        assert Path(directory, 'statement.tex').exists()
        self._source = LatexSource(Path(directory, 'statement.tex'))
        self._directory = directory
        self._num = num
        self._codename = codename

    @property
    def pdf(self) -> Optional[Path]:
        """Return path to pdf if exists."""
        return self._source.pdf

    def __str__(self) -> str:
        return str(self._source)

    @ui.work('LATEX')
    def build(self, blank_page: bool) -> ui.WorkResult:
        """Compile latex statement"""
        if self._num is not None:
            os.environ['OCIMATIC_PROBLEM_NUMBER'] = chr(ord('A') + self._num)
        if self._codename:
            os.environ['OCIMATIC_CODENAME'] = self._codename
        if blank_page:
            os.environ['OCIMATIC_BLANK_PAGE'] = 'True'

        success = self._source.compile() is not None

        return ui.WorkResult(success=success, short_msg='OK' if success else 'FAILED')

    def io_samples(self) -> List[Test]:
        """Find sample input data in the satement"""
        samples = set()
        for line in self._source.iter_lines():
            m = re.match(r'[^%]*\\sampleIO(?:\*)?(\[[^\]]*\]){0,2}{([^}]+)}', line)
            if m:
                samples.add(m.group(2))
        return [
            Test(Path(self._directory, f"{s}.in"), Path(self._directory, f"{s}.sol"))
            for s in samples
        ]

    def scores(self) -> List[int]:
        """Finds the scores for the subtasks"""
        scores = []
        for line in self._source.iter_lines():
            m = re.match(r'[^%]*\\subtask{([^}]+)}', line)
            if m:
                scores.append(int(m.group(1)))
        if not scores:
            ui.show_message('Warning', "Couldn't infer the score from the statement, assuming 100.",
                            ui.WARNING)
            scores = [100]

        return scores
