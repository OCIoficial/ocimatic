from __future__ import annotations

import glob
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pypdf
import tomlkit
from click.shell_completion import CompletionItem

import ocimatic
from ocimatic import utils
from ocimatic.checkers import Checker
from ocimatic.dataset import Dataset, RunMode, RuntimeStats, Test
from ocimatic.solutions import Solution
from ocimatic.source_code import CppSource, JavaSource, LatexSource, RustSource
from ocimatic.testplan import Testplan
from ocimatic.utils import Stn


def find_contest_root() -> tuple[Path, Path | None] | None:
    """Find the root contest's directory.

    Returns the absolute path to the root of the contest and the last directory
    before reaching the root, this correspond to the directory of the task in
    which ocimatic was called. If the function reach system root the program exists
    with an error.
    """
    last_dir = None
    curr_dir = Path.cwd()
    while not Path(curr_dir, ContestConfig.FILE_NAME).exists():
        last_dir = curr_dir
        curr_dir = curr_dir.parent
        if curr_dir.samefile(last_dir):
            return None
    ocimatic.config["contest_root"] = curr_dir
    return (curr_dir, last_dir)


@dataclass(kw_only=True, frozen=True)
class ContestConfig:
    FILE_NAME = "contest.toml"

    phase: str

    @staticmethod
    def init(contest_path: Path, phase: str) -> None:
        config_path = Path(contest_path, ContestConfig.FILE_NAME)
        with config_path.open("r+") as f:
            config = cast(dict[Any, Any], tomlkit.load(f))
            config["contest"]["phase"] = phase

            f.seek(0)
            tomlkit.dump(config, f)  # pyright: ignore [reportUnknownMemberType]
            f.truncate()

    @staticmethod
    def load(contest_path: Path) -> ContestConfig:
        config_path = Path(contest_path, ContestConfig.FILE_NAME)
        with config_path.open() as f:
            config = cast(dict[Any, Any], tomlkit.load(f))
            contest_table = config.get("contest", {})
            phase = contest_table.get("phase", "")
            return ContestConfig(phase=phase)


class Contest:
    """Represent a contest.

    A contest is formed by a list of tasks and a titlepage. A contest is associated
    to a directory in the filesystem.
    """

    COLOR = utils.MAGENTA

    @staticmethod
    def _detect_tasks_in(contest_dir: Path) -> list[tuple[TaskConfig, Path]]:
        tasks: list[tuple[TaskConfig, Path]] = []
        for dir in contest_dir.iterdir():
            config = TaskConfig.load(dir)
            if config:
                tasks.append((config, dir))
        tasks.sort()
        return tasks

    @staticmethod
    def select_task(contest_dir: Path, task_name: str) -> Task | None:
        tasks = Contest._detect_tasks_in(contest_dir)
        found = next(
            ((i, c, p) for i, (c, p) in enumerate(tasks) if c.codename == task_name),
            None,
        )
        if not found:
            return None
        return Task(found[2], found[1], found[0])

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._config = ContestConfig.load(directory)
        self._tasks = [
            Task(d, config, i)
            for (i, (config, d)) in enumerate(Contest._detect_tasks_in(directory))
        ]

        os.environ["OCIMATIC_PHASE"] = self._config.phase

        self._titlepage = LatexSource(Path(directory, "titlepage.tex"))

    @property
    def directory(self) -> Path:
        return self._directory

    @staticmethod
    def create_layout(dest: Path, phase: str | None) -> None:
        """Copy contest skeleton to `dest` and save configuration."""
        ocimatic_dir = Path(__file__).parent
        contest_skel = Path(ocimatic_dir, "resources", "contest-skel")
        shutil.copytree(
            contest_skel,
            dest,
            ignore=shutil.ignore_patterns("auto"),
            symlinks=True,
        )
        if phase is not None:
            ContestConfig.init(dest, phase)

    def new_task(self, name: str) -> None:
        task_dir = Path(self._directory, name)
        Task.create_layout(task_dir)

    @property
    def tasks(self) -> list[Task]:
        return self._tasks

    @utils.hd1("Generating problemset", color=COLOR)
    def build_problemset(self) -> utils.Status:
        """Build titlepage and statement of all tasks. Then merge all pdfs into a single pdf."""
        status = utils.Status.success
        status &= self._build_problemset_twoside()
        status &= self._build_problemset_oneside()
        return status

    @utils.hd1("oneside")
    def _build_problemset_oneside(self) -> utils.Status:
        os.environ["OCIMATIC_SIDENESS"] = "oneside"
        if self._compile_titlepage().is_fail():
            return utils.Status.fail

        status = utils.Status.success
        for task in self._tasks:
            status &= task.statement.build(blank_page=False).status

        if status == utils.Status.fail:
            return utils.Status.fail

        return self._merge_pdfs("oneside.pdf").status

    @utils.hd1("twoside")
    def _build_problemset_twoside(self) -> utils.Status:
        os.environ["OCIMATIC_SIDENESS"] = "twoside"
        if self._compile_titlepage().is_fail():
            return utils.Status.fail

        status: utils.Status = utils.Status.success
        for i, task in enumerate(self._tasks):
            last = i == len(self._tasks) - 1
            blank_page = last and ocimatic.config["last_blank_page"]
            status &= task.statement.build(blank_page=blank_page).status

        if status == utils.Status.fail:  # pyright: ignore [reportUnnecessaryComparison]
            return utils.Status.fail

        return self._merge_pdfs("twoside.pdf").status

    @utils.hd1("Creating archive", color=COLOR)
    def archive(self) -> None:
        """Package statements and datasets of all tasks into a single zip file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for task in self._tasks:
                if not task.copy_to(Path(tmpdir)):
                    utils.writeln()
                    utils.show_message(
                        "Error",
                        f"Couldn't copy task {task.name} to archive.",
                        utils.ERROR,
                    )
                    return

            self._build_problemset_twoside()
            twoside = Path(self._directory, "twoside.pdf")
            shutil.copy2(twoside, Path(tmpdir, "twoside.pdf"))

            self._build_problemset_oneside()
            oneside = Path(self._directory, "oneside.pdf")
            shutil.copy2(oneside, Path(tmpdir, "oneside.pdf"))

            shutil.make_archive("archive", "zip", tmpdir)

    @utils.work("LATEX", "titlepage.tex")
    def _compile_titlepage(self) -> utils.Result:
        """Compile title page latex."""
        result = self._titlepage.compile()
        if isinstance(result, Path):
            return utils.Result.success(short_msg="OK")
        else:
            return utils.Result.fail(short_msg="FAILED", long_msg=result.msg)

    @utils.work("MERGE", "{1}")
    def _merge_pdfs(self, filename: str) -> utils.Result:
        """Merge titlepage and statements pdfs into a single file."""
        try:
            merger = pypdf.PdfWriter()
            for task in self._tasks:
                if not task.statement.pdf:
                    return utils.Result.fail(
                        short_msg="FAILED",
                        long_msg="No statement",
                    )
                merger.append(task.statement.pdf)
            titlepage = Path(self._directory, "titlepage.pdf")
            if titlepage.exists():
                merger.append(titlepage)

            merger.write(Path(self._directory, filename))  # pyright: ignore [reportUnknownMemberType]
            merger.close()
            return utils.Result.success(short_msg="OK")
        except Exception as exc:
            return utils.Result.fail(short_msg="FAILED", long_msg=str(exc))

    @property
    def name(self) -> str:
        """Name of the contest."""
        return self._directory.name

    def __str__(self) -> str:
        return self.name

    def find_task(self, name: str) -> Task | None:
        """Find task with the given name."""
        return next((p for p in self._tasks if p.name == name), None)


@dataclass(kw_only=True, frozen=True)
class TaskConfig:
    FILE_NAME = "task.toml"

    codename: str
    priority: int
    static_dataset: bool

    @staticmethod
    def init(task_path: Path) -> None:
        config_path = Path(task_path, TaskConfig.FILE_NAME)
        with config_path.open("r+") as f:
            config = cast(dict[Any, Any], tomlkit.load(f))
            config["task"]["codename"] = task_path.name

            f.seek(0)
            tomlkit.dump(config, f)  # pyright: ignore [reportUnknownMemberType]
            f.truncate()

    @staticmethod
    def load(task_path: Path) -> TaskConfig | None:
        config_path = Path(task_path, TaskConfig.FILE_NAME)
        if not config_path.exists():
            return None

        with config_path.open() as f:
            config = cast(dict[Any, Any], tomlkit.load(f))
            task_table = config.get("task", {})
            codename = task_table.get("codename", task_path.name)
            priority = task_table.get("priority", 0)

            dataset_table = config.get("dataset", {})
            static_dataset = dataset_table.get("static", False)
            return TaskConfig(
                codename=codename,
                priority=priority,
                static_dataset=static_dataset,
            )

    def __lt__(self, other: TaskConfig) -> bool:
        return (self.priority, self.codename) < (other.priority, other.codename)


class Task:
    """Represent a task.

    A task consists of a statement, a list of correct and partial solutions,
    and a dataset. A task is associated to a directory in the filesystem.
    """

    COLOR = utils.MAGENTA + utils.BOLD

    @staticmethod
    def create_layout(task_path: Path) -> None:
        ocimatic_dir = Path(__file__).parent

        # Copy task skeleton
        task_skel = Path(ocimatic_dir, "resources", "task-skel")
        shutil.copytree(task_skel, task_path, symlinks=True)

        # Init config
        TaskConfig.init(task_path)

        # We put oci.cls and logo.eps in the statement directory to make it easier to work on the
        # pdf without using ocimatic.
        contest_skel = Path(ocimatic_dir, "resources", "contest-skel")
        statement_path = Path(task_path, "statement")
        shutil.copy(Path(contest_skel, "oci.cls"), Path(statement_path, "oci.cls"))
        shutil.copy(Path(contest_skel, "logo.eps"), Path(statement_path, "logo.eps"))

    def __init__(self, directory: Path, config: TaskConfig, num: int) -> None:
        self._directory = directory
        self._config = config

        self._managers_dir = Path(directory, "managers")

        self._checker = Checker.find_in_directory(self._managers_dir)

        correct_dir = Path(directory, "solutions", "correct")
        self._correct = Solution.load_solutions_in_dir(
            self.codename,
            correct_dir,
            self._managers_dir,
        )
        partial_dir = Path(directory, "solutions", "partial")
        self._partial = Solution.load_solutions_in_dir(
            self.codename,
            partial_dir,
            self._managers_dir,
        )

        self._statement = Statement(
            Path(directory, "statement"),
            num=num,
            codename=self.codename,
        )

        self._testplan = None
        if not self._config.static_dataset:
            self._testplan = Testplan(
                Path(self._directory, "testplan"),
                self._directory,
                Path(self._directory, "dataset"),
            )

        self._dataset = Dataset(
            Path(directory, "dataset"),
            self._testplan,
            self._statement.io_samples(),
        )

    @property
    def codename(self) -> str:
        return self._config.codename

    @utils.hd1("{0}", "Copy to archive")
    def copy_to(self, directory: Path) -> bool:
        new_dir = Path(directory, self.codename)
        new_dir.mkdir()

        if self._dataset.compress(random_sort=False).is_fail():
            return False

        dataset = Path(self._directory, "dataset", "data.zip")
        dataset_dst = Path(new_dir, "data.zip")
        shutil.copy2(dataset, dataset_dst)

        if self.statement.build(blank_page=False).is_fail():
            return False

        statement = Path(new_dir, "statement.pdf")
        shutil.copy2(Path(self._directory, "statement", "statement.pdf"), statement)
        return True

    @utils.hd1("{0}", "Running testplan", COLOR)
    def run_testplan(self, stn: Stn | None) -> utils.Status:
        if self._testplan is None:
            utils.fatal_error("Task has a static dataset.")
        return self._testplan.run(stn)

    def load_solution_from_path(self, path: Path) -> Solution | None:
        """Search for a solution matching a path.

        The behavior depends on whether the path is absolute or relative. If absolute,
        it will match a solution for the corresponding path. If it is relative, it will
        try to match the path relative to the following locations, in order, until it
        finds a match or it fails to find one:
        1. <task>/solutions/correct
        2. <task>/solutions/partial
        3. <task>/solutions/
        4. <cwd>
        Where <task> is the path of the current task and <cwd> is the current working
        directory.
        """
        if path.is_absolute():
            return Solution.load(self.codename, path, self._managers_dir)

        for dir in [
            Path(self._directory, "solutions", "correct"),
            Path(self._directory, "solutions", "partial"),
            Path(self._directory, "solutions"),
            Path.cwd(),
        ]:
            sol = Solution.load(self.codename, Path(dir, path), self._managers_dir)
            if sol:
                return sol
        return None

    def solution_completion(self, incomplete: str) -> list[CompletionItem]:
        candidates: dict[str, str] = {
            sol.source.file.name: "correct" for sol in self._correct
        }
        for sol in self._partial:
            key = sol.source.file.name
            if key in candidates:
                key = "partial" + os.path.sep + key
            candidates[key] = "partial"

        for key in glob.glob(incomplete + "*"):  # noqa: PTH207
            if key in candidates:
                continue
            path = Path(key)
            if path.is_dir():
                candidates[key + os.path.sep] = "directory"
            elif path.suffix in Solution.VALID_EXTENSIONS:
                candidates[key] = "file"

        completions = [
            CompletionItem(value=k, help=v)
            for k, v in candidates.items()
            if k.startswith(incomplete)
        ]
        return completions

    @utils.hd1("{0}", "Validating input files", COLOR)
    def validate_input(self, stn: Stn | None) -> utils.Status:
        return self._dataset.validate_input(stn)

    @utils.hd1("{0}", "Compressing dataset", COLOR)
    def compress_dataset(self, *, random_sort: bool) -> None:
        """Compress dataset into a single file."""
        self._dataset.compress(random_sort=random_sort)

    @property
    def name(self) -> str:
        """Name of the task."""
        return self._config.codename

    def __str__(self) -> str:
        return self.name

    @property
    def statement(self) -> Statement:
        return self._statement

    @utils.hd1("{0}", "Score Params", COLOR)
    def score(self) -> None:
        counts = self._dataset.count()
        scores = self._statement.scores()
        if len(scores) != len(counts):
            utils.show_message(
                "Error",
                "The number of subtasks in the statement doesn't match the number of "
                "subtasks in the dataset.",
                utils.ERROR,
            )
            return

        if len(counts) == len(scores) == 1:
            utils.show_message("Sum", str(scores[0] / counts[0]))
        utils.show_message(
            "GroupMin",
            str([[m, t] for (m, t) in zip(scores, counts, strict=True)]),
        )

    @utils.hd1("{0}", "Normalizing", COLOR)
    def normalize(self) -> None:
        self._dataset.normalize()

    @utils.hd1("{0}", "Running solution", COLOR)
    def run_solution(self, solution: Path, timeout: float, stn: Stn | None) -> None:
        """Run all solutions reporting outcome and running time."""
        sol = self.load_solution_from_path(solution)
        if not sol:
            return utils.show_message("Error", "Solution not found", utils.ERROR)

        results = sol.run_on_dataset(
            self._dataset,
            self._checker,
            RunMode.run_solution,
            timeout=timeout,
            stn=stn,
        )
        if results:
            utils.writeln()
            stats = results.runtime_stats()
            if stats:
                _write_stats(stats)

            if sol.is_partial:
                should_fail = ", ".join(
                    f"st{st}" for st in sorted(sol.should_fail(results))
                )
                if sol.check_results(results):
                    utils.writeln()
                    utils.writeln(
                        f"Solution failed the subtasks it was supposed to: should-fail=[{should_fail}]",
                        utils.OK,
                    )
                else:
                    failed = ", ".join(
                        f"st{stn}" for stn in sorted(results.failed_subtasks())
                    )
                    utils.write(
                        f"""
The results don't match the solution's specification.
 - Subtasks expected to fail: [{should_fail}]
 - Subtasks that failed: [{failed}]

To specify which subtasks the solution should pass/fail, you must either have a `should-pass`
or `should-fail` comment at the beginning of the file. For example, to specify that a solution
should pass subtasks 1 and 2, write the following comment at the beginning of the file:
// @ocimatic should-pass=[st1, st2]
If no comment is specified, ocimatic will assume that all subtasks should fail.
""",
                        utils.ERROR,
                    )
            else:
                utils.writeln()
                if results.check_all_correct():
                    utils.writeln("Result: All test passed", utils.OK)
                else:
                    utils.writeln("Result: Some test failed", utils.ERROR)

    @utils.hd1("{0}", "Checking dataset", COLOR)
    def check_dataset(self) -> utils.Status:
        """Check input/output correctness.

        First run all correct solutions againt all test cases and sample input. Then use the running
        time of correct solutions to set a timeout. Finally, use the timeout to run partial solutions
        and ensure they fail the subtasks they are suppose to fail.
        """
        if sum(self._dataset.count()) == 0:
            utils.show_message(
                "Error",
                "No test cases found. Generate the dataset by running `ocimatic run-testplan && ocimatic gen-expected`.",
                utils.ERROR,
            )
            return utils.Status.fail

        if not self._dataset.check_all_have_expected():
            utils.show_message(
                "Error",
                "Some test cases don't have expected output, generate them with `ocimatic gen-expected`.",
                utils.ERROR,
            )
            return utils.Status.fail

        # Do not early return if there are validate_input errors but still report it at the end
        validate_input_status = self._check_dataset_validate_input()

        stats = self._check_dataset_run_correct_solutions()
        if not stats:
            return utils.Status.fail

        return (
            self._check_dataset_run_partial_solutions(stats) and validate_input_status
        )

    def _check_dataset_validate_input(self) -> utils.Status:
        utils.writeln()
        utils.writeln("Running validators on input files", utils.INFO)
        utils.writeln()
        status = self._dataset.validate_input(stn=None)
        utils.writeln()
        if status == utils.Status.fail:
            utils.writeln()
            utils.writeln("Some subtasks didn't pass input validation.", utils.ERROR)

        return status

    def _check_dataset_run_correct_solutions(self) -> RuntimeStats | None:
        stats = RuntimeStats.unit()

        if not self._correct:
            utils.show_message(
                "Error",
                "at least one correct solution needed",
                utils.ERROR,
            )
            return None

        # Run correct solutions
        utils.writeln("Running correct solutions", utils.INFO)
        failed: list[Solution] = []
        for sol in self._correct:
            results = sol.run_on_dataset(
                self._dataset,
                self._checker,
                RunMode.check_correct,
            )
            if results is None or not results.check_all_correct():
                failed.append(sol)
                continue
            new_stats = results.runtime_stats()
            assert new_stats
            stats += new_stats

        if failed:
            utils.write(
                """
Summary
-------
Some correct solutions failed to run or produced wrong results. Run them individually with
`ocimatic run` to get more information.

Solutions with issues:
""",
                utils.RED,
            )

            for sol in failed:
                utils.writeln(" * " + str(sol), utils.RED)
            return None

        utils.writeln()
        _write_stats(stats)
        utils.writeln()
        utils.writeln("All correct solutions produced correct results", utils.GREEN)
        return stats

    def _check_dataset_run_partial_solutions(self, stats: RuntimeStats) -> utils.Status:
        timeout = stats.set_limit()

        utils.writeln()
        utils.writeln("Running partial solutions", utils.INFO)
        utils.writeln()
        utils.writeln(
            f"Timeout set to {timeout:.1f}s ({stats.fmt_limit_calculation()})",
        )
        if not self._partial:
            utils.writeln()
            utils.writeln("No partial solutions", utils.WARNING)
            return utils.Status.success

        failed: list[Solution] = []
        for sol in self._partial:
            results = sol.run_on_dataset(
                self._dataset,
                self._checker,
                RunMode.check_partial,
                timeout=timeout,
            )
            if results is None or not sol.check_results(results):
                if len(self._partial) > 1:
                    utils.writeln("issues found", utils.RED)
                failed.append(sol)

        if failed:
            utils.write(
                """
Summary
-------
Some partial solutions had issues when running or didn't pass/fail the subtasks they were supposed to.
Run them individually with `ocimatic run` to get more information. Remember to set an appropiate timeout
using the `--timeout` flag.

Solutions with issues:
""",
                utils.RED,
            )
            for sol in failed:
                utils.writeln(" * " + str(sol), utils.RED)
            return utils.Status.fail

        utils.writeln()
        utils.writeln(
            "All partial solutions passed/failed the subtasks they were supposed to.",
            utils.GREEN,
        )
        return utils.Status.success

    @utils.hd1("{0}", "Building solutions", COLOR)
    def build_solution(self, solution: Path) -> None:
        """Force compilation of solutions."""
        sol = self.load_solution_from_path(solution)
        if not sol:
            return utils.show_message("Error", "Solution not found", utils.ERROR)
        sol.build()

    @utils.hd1("{0}", "Generating expected output", COLOR)
    def gen_expected(
        self,
        *,
        sample: bool = False,
        solution: Path | None = None,
    ) -> utils.Status:
        """Generate expected outputs files for the dataset by running a correct solution.

        If `sample` is True, also generate expected output for sample input. If `solution` is
        not `None` use it to generate the expected output, otherwise use any correct one
        prioritizing C++ solutions.
        """
        if self._config.static_dataset:
            utils.show_message("Skipping", "Task has a static dataset.", utils.WARNING)
            return utils.Status.success
        if not self._correct:
            utils.show_message("Error", "No correct solution.", utils.ERROR)
            return utils.Status.fail
        generator = None
        if solution:
            generator = self.load_solution_from_path(solution)
        else:
            keys: dict[type, int] = {CppSource: 0, RustSource: 1, JavaSource: 2}
            sols = sorted(
                self._correct,
                key=lambda sol: keys.get(
                    type(sol.source),
                    len(keys),
                ),
            )
            generator = sols[0] if sols else None

        if not generator:
            utils.fatal_error("solution not found")
        if generator.gen_expected(self._dataset, sample=sample) == utils.Status.fail:
            return utils.Status.fail

        if sum(self._dataset.count()) == 0:
            utils.show_message("Warning", "Empty dataset", utils.WARNING)

        return utils.Status.success

    @utils.hd1("{0}", "Building statement", COLOR)
    def build_statement(self, *, blank_page: bool = False) -> None:
        """Generate pdf for the statement."""
        self._statement.build(blank_page=blank_page)


class Statement:
    """Represents a statement. A statement is composed by a latex source and a pdf file."""

    def __init__(
        self,
        directory: Path,
        num: int | None = None,
        codename: str | None = None,
    ) -> None:
        assert Path(directory, "statement.tex").exists()
        self._source = LatexSource(Path(directory, "statement.tex"))
        self._directory = directory
        self._num = num
        self._codename = codename

    @property
    def pdf(self) -> Path | None:
        """Return path to pdf if exists."""
        return self._source.pdf

    def __str__(self) -> str:
        return str(self._source)

    @utils.work("LATEX")
    def build(self, *, blank_page: bool) -> utils.Result:
        """Compile latex statement."""
        if self._num is not None:
            os.environ["OCIMATIC_PROBLEM_NUMBER"] = chr(ord("A") + self._num)
        if self._codename:
            os.environ["OCIMATIC_CODENAME"] = self._codename
        if blank_page:
            os.environ["OCIMATIC_BLANK_PAGE"] = "True"

        result = self._source.compile()
        if isinstance(result, Path):
            return utils.Result.success("OK")
        else:
            return utils.Result.fail("FAILED", long_msg=result.msg)

    def io_samples(self) -> list[Test]:
        """Find sample input data in the satement."""
        samples: set[str] = set()
        for line in self._source.iter_lines():
            m = re.match(r"[^%]*\\sampleIO(?:\*)?(\[[^\]]*\]){0,2}{([^}]+)}", line)
            if m:
                samples.add(m.group(2))
        return [
            Test(Path(self._directory, f"{s}.in"), Path(self._directory, f"{s}.sol"))
            for s in samples
        ]

    def scores(self) -> list[int]:
        """Find the scores for the subtasks."""
        scores: list[int] = []
        for line in self._source.iter_lines():
            m = re.match(r"[^%]*\\subtask{([^}]+)}", line)
            if m:
                scores.append(int(m.group(1)))
        if not scores:
            utils.show_message(
                "Warning",
                "Couldn't infer the score from the statement, assuming 100.",
                utils.WARNING,
            )
            scores = [100]

        return scores


def _write_stats(stats: RuntimeStats) -> None:
    utils.writeln("Running time")
    utils.writeln(f"  Max: {stats.max:.3f}s")
    utils.writeln(f"  Min: {stats.min:.3f}s")
