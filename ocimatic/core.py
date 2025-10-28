from __future__ import annotations

import glob
import json
import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from enum import Enum, StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, cast

import msgspec
import tomlkit
from click.shell_completion import CompletionItem

from ocimatic import config, ui
from ocimatic.checkers import Checker
from ocimatic.dataset import Dataset, RunMode, RuntimeStats, Test
from ocimatic.result import Result, Status, Error
from ocimatic.solutions import Solution
from ocimatic.source_code import (
    CppSource,
    JavaSource,
    LatexSource,
    PDFSource,
    RustSource,
    TypstSource,
)
from ocimatic.testplan import Testplan
from ocimatic.utils import SortedDict, Stn


class Typesetting(StrEnum):
    TYPST = "typst"
    LATEX = "latex"


class CLI:
    def __init__(self) -> None:
        self._data: tuple[Contest, Path | None] | None = None

    @staticmethod
    def find_contest_root() -> tuple[Path, Path | None] | None:
        """Find the root of the contest.

        Returns the absolute path to the root of the contest and the last directory
        before reaching the root (if there's one), this is used to find the target
        task. Returns `None` if the function reaches the system root without finding
        a contest.
        """
        last_dir = None
        curr_dir = Path.cwd()
        while not Path(curr_dir, ContestConfig.FILE_NAME).exists():
            last_dir = curr_dir
            curr_dir = curr_dir.parent
            if curr_dir.samefile(last_dir):
                return None
        config.CONTEST_ROOT = curr_dir
        return (curr_dir, last_dir)

    @staticmethod
    def load_task_by_name(contest_dir: Path, task_name: str) -> Task | None:
        return Contest.load_task_by_name(contest_dir, task_name)

    @staticmethod
    def load_task_by_dir(contest_dir: Path, task_dir: Path) -> Task | None:
        return Contest.load_task_by_dir(contest_dir, task_dir)

    @staticmethod
    def init_contest(dest: Path, phase: str, typesetting: Typesetting) -> None:
        Contest.create_layout(dest, phase, typesetting)

    @property
    def contest(self) -> Contest:
        (contest, _) = self._load()
        return contest

    @property
    def last_dir(self) -> Path | None:
        (_, last_dir) = self._load()
        return last_dir

    def _load(self) -> tuple[Contest, Path | None]:
        if not self._data:
            result = CLI.find_contest_root()
            if not result:
                ui.fatal_error("ocimatic was not called inside a contest.")
            self._data = (Contest(result[0]), result[1])
        return self._data

    def new_task(self, name: str) -> None:
        if Path(self.contest.directory, name).exists():
            ui.fatal_error("Cannot create task in existing directory.")
        self.contest.new_task(name)
        ui.show_message("Info", f"Task [{name}] created", ui.OK)

    def select_task(self, name: str | None) -> Task | None:
        task = None
        if name is not None:
            task = self.contest.find_task_by_name(name)
        elif self.last_dir:
            task = self.contest.find_task_by_dir(self.last_dir)
        return task

    def select_tasks(self) -> list[Task]:
        task = None
        if self.last_dir:
            task = self.contest.find_task_by_dir(self.last_dir)
        if task is not None:
            return [task]
        else:
            return self.contest.tasks


class ContestConfig(msgspec.Struct, kw_only=True, frozen=True):
    class Contest(msgspec.Struct, kw_only=True, frozen=True):
        phase: str
        typesetting: Typesetting

    FILE_NAME = "contest.toml"

    contest: ContestConfig.Contest

    @staticmethod
    def init(contest_path: Path, phase: str | None, typesetting: Typesetting) -> None:
        conf_path = Path(contest_path, ContestConfig.FILE_NAME)
        with conf_path.open("r+") as f:
            conf = cast(dict[Any, Any], tomlkit.load(f))
            if phase is not None:
                conf["contest"]["phase"] = phase
            conf["contest"]["typesetting"] = typesetting

            f.seek(0)
            tomlkit.dump(conf, f)  # pyright: ignore [reportUnknownMemberType]
            f.truncate()

    @staticmethod
    def load(contest_path: Path) -> ContestConfig:
        path = Path(contest_path, ContestConfig.FILE_NAME)
        try:
            conf = msgspec.toml.decode(path.read_text(), type=ContestConfig)
        except Exception as e:
            ui.fatal_error(f"Failed to load contest config from {path}: {e}")
        return conf


class Resource:
    """A resource is a file or directory in ocimatic that's copied when creating a contest or a task.

    Some resources can be synchronized after the contest or task has been created. This is useful
    when developing or fixing bugs in ocimatic if a contest has already been initialized.
    """

    def __init__(self, *args: str, root: bool = False, sync: bool = False) -> None:
        assert not (root and sync), "A root resource cannot be sync"
        ocimatic_dir = Path(__file__).parent
        resources_dir = ocimatic_dir / "resources"
        self._path = resources_dir / Path(*args)
        self._sync = sync
        self._root = root

    def copy_to(self, dest: Path) -> None:
        if self._path.is_dir():
            dest = dest if self._root else dest / self._path.name
            shutil.copytree(
                self._path,
                dest,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("auto"),
                symlinks=True,
            )
        else:
            shutil.copy2(self._path, dest)

    @ui.work("SYNC")
    def sync(self, dest: Path) -> Result:
        if not self._sync:
            return Result.success("SKIPPED")
        self.copy_to(dest)
        return Result.success("OK")

    def __str__(self) -> str:
        return str(self._path)


class Contest:
    """Represent a contest.

    A contest is formed by a list of tasks and a titlepage. A contest is associated
    to a directory in the filesystem.
    """

    COLOR = ui.MAGENTA

    SKEL = Resource("contest-skel", root=True)

    RESOURCES: ClassVar[list[Resource]] = [Resource(".github", sync=True)]

    TYPESETTING_RESOURCES: ClassVar[dict[Typesetting, list[Resource]]] = {
        Typesetting.LATEX: [
            Resource("latex", "logo.eps", sync=True),
            Resource("latex", "oci.cls", sync=True),
            Resource("latex", "titlepage.tex"),
            Resource("latex", "general.tex", sync=True),
        ],
        Typesetting.TYPST: [
            Resource("typst", "logo.png", sync=True),
            Resource("typst", "oci.typ", sync=True),
            Resource("typst", "titlepage.typ"),
            Resource("typst", "general.typ", sync=True),
            Resource("typst", "fonts", sync=True),
        ],
    }

    @staticmethod
    def create_layout(dest: Path, phase: str, typesetting: Typesetting) -> None:
        """Copy contest skeleton to `dest`."""
        Contest.SKEL.copy_to(dest)
        for resource in Contest.RESOURCES:
            resource.copy_to(dest)
        for resource in Contest.TYPESETTING_RESOURCES[typesetting]:
            resource.copy_to(dest)
        ContestConfig.init(dest, phase, typesetting)

    @staticmethod
    def _detect_tasks(contest_dir: Path) -> Iterator[tuple[int, TaskConfig, Path]]:
        tasks: list[tuple[TaskConfig, Path]] = []
        for dir in contest_dir.iterdir():
            conf = TaskConfig.load(dir)
            if conf:
                tasks.append((conf, dir))
        tasks.sort()
        return ((i, c, d) for i, (c, d) in enumerate(tasks))

    @staticmethod
    def load_task_by_name(contest_dir: Path, task_name: str) -> Task | None:
        contest_conf = ContestConfig.load(contest_dir)
        return next(
            (
                Task(d, contest_conf, conf, i)
                for i, conf, d in Contest._detect_tasks(contest_dir)
                if conf.task.codename == task_name
            ),
            None,
        )

    @staticmethod
    def load_task_by_dir(contest_dir: Path, task_dir: Path) -> Task | None:
        contest_conf = ContestConfig.load(contest_dir)
        return next(
            (
                Task(d, contest_conf, conf, i)
                for i, conf, d in Contest._detect_tasks(contest_dir)
                if d == task_dir
            ),
            None,
        )

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._conf = ContestConfig.load(directory)
        self._tasks = [
            Task(d, self._conf, conf, i)
            for i, conf, d in Contest._detect_tasks(directory)
        ]

        vars = {"OCIMATIC_PHASE": self._conf.contest.phase}
        self._titlepage: PDFSource
        self._general: PDFSource
        match self._conf.contest.typesetting:
            case Typesetting.LATEX:
                self._titlepage = LatexSource(directory / "titlepage.tex", env=vars)
                self._general = LatexSource(directory / "general.tex", env=vars)
            case Typesetting.TYPST:
                self._titlepage = TypstSource(
                    directory / "titlepage.typ",
                    sys_inputs=vars,
                )
                self._general = TypstSource(
                    directory / "general.typ",
                    sys_inputs=vars,
                )

    @property
    def directory(self) -> Path:
        return self._directory

    def new_task(self, name: str) -> None:
        Task.create_layout(self._directory / name, self._conf.contest.typesetting)

    @property
    def tasks(self) -> list[Task]:
        return self._tasks

    @ui.hd1("Generating problemset", color=COLOR)
    def build_problemset(self) -> Status:
        """Build titlepage and statement for all tasks. Then merge all pdfs into a single pdf."""
        return self._build_problemset()

    def _build_problemset(self) -> Status:
        if self._titlepage.compile_work().is_fail():
            return Status.fail

        if self._general.compile_work().is_fail():
            return Status.fail

        status = Status.success
        for task in self._tasks:
            status &= task.statement.build().status

        status &= self._merge_pdfs(Sideness.ONESIDE).status
        status &= self._merge_pdfs(Sideness.TWOSIDE).status

        return status

    @ui.work("MERGE", "{1}.pdf")
    def _merge_pdfs(self, sideness: Sideness) -> Result:
        """Merge titlepage and statements pdfs into a single file."""
        # This is slow to import so we do it lazily
        from pypdf import PdfWriter

        def _add_blank_page(merger: PdfWriter, side: Sideness, even: Evenness) -> None:
            if side == Sideness.TWOSIDE and even.check(len(merger.pages)):
                merger.add_blank_page()

        try:
            merger = PdfWriter()
            titlepage = self._directory / "titlepage.pdf"
            general = self._directory / "general.pdf"
            if titlepage.exists():
                merger.append(titlepage, import_outline=False)
                _add_blank_page(merger, sideness, Evenness.ODD)
            if general.exists():
                merger.append(
                    general,
                    outline_item="Información General",
                    import_outline=False,
                )
                _add_blank_page(merger, sideness, Evenness.ODD)
            for task in self._tasks:
                if not task.statement.pdf:
                    return Result.fail(
                        short_msg="FAILED",
                        long_msg="No statement",
                    )
                merger.append(
                    task.statement.pdf,
                    outline_item=task.title,
                    import_outline=False,
                )
                _add_blank_page(merger, sideness, Evenness.ODD)

            _add_blank_page(merger, sideness, Evenness.EVEN)
            merger.write(self._directory / f"{sideness}.pdf")
            merger.close()
            return Result.success(short_msg="OK")
        except Exception as exc:
            return Result.fail(short_msg="FAILED", long_msg=str(exc))

    @ui.hd1("Creating archive", color=COLOR)
    def archive(self) -> None:
        """Package statements and datasets of all tasks into a single zip file."""
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            for task in self._tasks:
                if not task.copy_to(tmpdir):
                    ui.writeln()
                    ui.show_message(
                        "Error",
                        f"Couldn't copy task {task.name} to archive.",
                        ui.ERROR,
                    )
                    return

            self._archive_problemset(tmpdir)

            Path("archive.zip").unlink(missing_ok=True)
            shutil.make_archive("archive", "zip", tmpdir)

    @ui.hd1("Problemset", "Copy to archive")
    def _archive_problemset(self, dest: Path) -> None:
        self._build_problemset()
        shutil.copy2(self._directory / f"{Sideness.TWOSIDE}.pdf", dest)
        shutil.copy2(self._directory / f"{Sideness.ONESIDE}.pdf", dest)

    @property
    def name(self) -> str:
        """Name of the contest."""
        return self._directory.name

    def __str__(self) -> str:
        return self.name

    def find_task_by_dir(self, dir: Path) -> Task | None:
        for task in self._tasks:
            if task.directory == dir:
                return task
        return None

    def find_task_by_name(self, name: str) -> Task | None:
        """Find task with the given name."""
        return next((p for p in self._tasks if p.name == name), None)

    @ui.hd1("{0}", "Sync", COLOR)
    def sync_resources(self) -> None:
        typesetting = self._conf.contest.typesetting
        for resource in Contest.RESOURCES:
            resource.sync(self._directory)
        for resource in Contest.TYPESETTING_RESOURCES[typesetting]:
            resource.sync(self._directory)
        for task in self._tasks:
            task.sync_resources(typesetting)


class TaskConfig(msgspec.Struct, kw_only=True, frozen=True):
    FILE_NAME = "task.toml"

    class Task(msgspec.Struct, kw_only=True, frozen=True):
        codename: str
        priority: int = 0

    class Dataset(msgspec.Struct, kw_only=True, frozen=True):
        static: bool = False

    task: TaskConfig.Task
    dataset: TaskConfig.Dataset

    @staticmethod
    def init(task_path: Path) -> None:
        conf_path = Path(task_path, TaskConfig.FILE_NAME)
        with conf_path.open("r+") as f:
            conf = cast(dict[Any, Any], tomlkit.load(f))
            conf["task"]["codename"] = task_path.name

            f.seek(0)
            tomlkit.dump(conf, f)  # pyright: ignore [reportUnknownMemberType]
            f.truncate()

    @staticmethod
    def load(task_path: Path) -> TaskConfig | None:
        path = Path(task_path, TaskConfig.FILE_NAME)
        if not path.exists():
            return None

        try:
            conf = msgspec.toml.decode(path.read_text(), type=TaskConfig)
        except Exception as e:
            ui.fatal_error(f"Failed to load task config from {path}: {e}")
        return conf

    def __lt__(self, other: TaskConfig) -> bool:
        return (self.task.priority, self.task.codename) < (
            other.task.priority,
            other.task.codename,
        )


class Task:
    """Represent a task.

    A task consists of a statement, a list of correct and partial solutions,
    and a dataset. A task is associated to a directory in the filesystem.
    """

    COLOR = ui.MAGENTA + ui.BOLD

    SKEL = Resource("task-skel", root=True)

    STATEMENT_RESOURCES: ClassVar[dict[Typesetting, list[Resource]]] = {
        Typesetting.LATEX: [
            Resource("latex", "statement.tex"),
            Resource("latex", "logo.eps", sync=True),
            Resource("latex", "oci.cls", sync=True),
        ],
        Typesetting.TYPST: [
            Resource("typst", "statement.typ"),
            Resource("typst", "logo.png", sync=True),
            Resource("typst", "oci.typ", sync=True),
            Resource("typst", "fonts", sync=True),
        ],
    }

    @staticmethod
    def create_layout(task_path: Path, typesetting: Typesetting) -> None:
        # Copy resources
        Task.SKEL.copy_to(task_path)
        for resource in Task.STATEMENT_RESOURCES[typesetting]:
            resource.copy_to(task_path / "statement")

        # Init config
        TaskConfig.init(task_path)

    def __init__(
        self,
        directory: Path,
        contest_conf: ContestConfig,
        conf: TaskConfig,
        num: int,
    ) -> None:
        self._directory = directory
        self._conf = conf

        self._managers_dir = directory / "managers"

        self._checker = Checker.find_in_directory(self._managers_dir)

        self._correct = Solution.load_solutions_in_dir(
            self.codename,
            directory / "solutions" / "correct",
            self._managers_dir,
        )
        self._partial = Solution.load_solutions_in_dir(
            self.codename,
            directory / "solutions" / "partial",
            self._managers_dir,
        )

        self._statement = Statement.load(
            directory / "statement",
            contest_conf.contest.typesetting,
            phase=contest_conf.contest.phase,
            num=num,
            codename=self.codename,
        )

        testplan = None
        if not self._conf.dataset.static:
            testplan = Testplan(
                directory / "testplan",
                directory,
                directory / "dataset",
            )

        self._dataset = Dataset(
            directory / "dataset",
            testplan,
            self._statement.get_io_samples(),
        )

    @property
    def codename(self) -> str:
        return self._conf.task.codename

    @property
    def directory(self) -> Path:
        return self._directory

    @cached_property
    def title(self) -> str:
        return self._statement.get_title()

    @ui.hd1("{0}", "Copy to archive")
    def copy_to(self, directory: Path) -> bool:
        new_dir = directory / self.codename
        new_dir.mkdir()

        if self._dataset.compress(random_sort=False).is_fail():
            return False

        shutil.copy2(self._directory / "dataset" / "data.zip", new_dir / "data.zip")

        if self.statement.build().is_fail():
            return False

        with (new_dir / "data.txt").open("w") as f:
            for st, regex in self._dataset.regexes().items():
                f.write(f"Subtask {st}: {regex}\n")

        statement = new_dir / "statement.pdf"
        shutil.copy2(self._directory / "statement" / "statement.pdf", statement)
        return True

    @ui.hd1("{0}", "Running testplan", COLOR)
    def run_testplan(self, stn: Stn | None) -> Status:
        if self._dataset.testplan is None:
            ui.writeln(" Skipping: task has static dataset")
            return Status.success
        return self._dataset.testplan.run(stn)

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
            self._directory / "solutions" / "correct",
            self._directory / "solutions" / "partial",
            self._directory / "solutions",
            Path.cwd(),
        ]:
            sol = Solution.load(self.codename, dir / path, self._managers_dir)
            if sol:
                return sol
        return None

    def solution_completion(
        self,
        incomplete: str,
        *,
        partial: bool,
    ) -> list[CompletionItem]:
        candidates: dict[str, str] = {
            sol.source.file.name: "correct" for sol in self._correct
        }
        if partial:
            for sol in self._partial:
                key = sol.source.file.name
                if key in candidates:
                    key = "partial" + os.path.sep + key
                candidates[key] = "partial"

            for kind, sols in [("correct", self._correct), ("partial", self._partial)]:
                if incomplete.startswith(kind):
                    for sol in sols:
                        key = kind + os.path.sep + sol.source.file.name
                        if key not in candidates:
                            candidates[key] = kind
                else:
                    candidates[kind + os.path.sep] = "group"

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

    @ui.hd1("{0}", "Validating input files", COLOR)
    def validate_input(self, stn: Stn | None) -> Status:
        return self._dataset.validate_input(stn)

    @ui.hd1("{0}", "Validating output files", COLOR)
    def validate_output(self, stn: Stn | None) -> Status:
        return self._dataset.validate_output(stn)

    @ui.hd1("{0}", "Compressing dataset", COLOR)
    def compress_dataset(self, *, random_sort: bool) -> None:
        """Compress dataset into a single file."""
        self._dataset.compress(random_sort=random_sort)

    @property
    def name(self) -> str:
        """Name of the task."""
        return self._conf.task.codename

    def __str__(self) -> str:
        return self.name

    @property
    def statement(self) -> Statement:
        return self._statement

    @ui.hd1("{0}", "Score Params", COLOR)
    def score_params(self) -> None:
        counts = self._dataset.counts()
        scores = self._statement.get_scores()
        regexes = self._dataset.regexes()
        assert regexes.keys() == counts.keys()
        if scores.keys() != counts.keys():
            ui.show_message(
                "error",
                "the number of subtasks in the statement doesn't match the number of "
                "subtasks in the dataset.",
                ui.ERROR,
            )
            return

        if len(counts) == len(scores) == 1:
            ui.show_message("Sum", str(scores[Stn(1)] / counts[Stn(1)]))
        ui.show_message(
            "GroupMin",
            json.dumps(
                [
                    [m, f"{regex}"]
                    for (m, regex) in zip(
                        scores.values(),
                        regexes.values(),
                        strict=True,
                    )
                ],
            ),
        )

    @ui.hd1("{0}", "Solutions", COLOR)
    def list_solutions(self) -> None:
        for sol in self._correct:
            ui.writeln(f" * [correct] {sol.source.file.name}", ui.CYAN)
        for sol in self._partial:
            ui.write(f" * [partial] {sol.source.file.name}", ui.CYAN)
            if isinstance(expected := sol.load_expected_outcome(self._dataset), Error):
                ui.writeln("  error in expected outcome comment", ui.ERROR)
            else:
                fmt = ", ".join(f"{k!r}={v}" for k, v in expected.items())
                ui.writeln(f" [{fmt}]", ui.CYAN)

    @ui.hd1("{0}", "Coverage", COLOR)
    def coverage(self) -> None:
        for sol in self._correct:
            sol.coverage(self._dataset)

    @ui.hd1("{0}", "Normalizing", COLOR)
    def normalize(self) -> None:
        self._dataset.normalize()

    @ui.hd1("{0}", "Running solution", COLOR)
    def run_solution(self, solution: Path, timeout: float, stn: Stn | None) -> None:
        """Run a solution reporting outcome and running time."""
        sol = self.load_solution_from_path(solution)
        if not sol:
            return ui.show_message("Error", "Solution not found", ui.ERROR)

        if stn is not None:
            subtask_results = sol.run_on_subtask(
                self._dataset,
                self._checker,
                timeout=timeout,
                stn=stn,
            )
            if not subtask_results:
                return
            if stats := subtask_results.runtime_stats():
                ui.writeln()
                _write_stats(stats)
        else:
            dataset_results = sol.run_on_dataset(
                self._dataset,
                self._checker,
                RunMode.run_solution,
                timeout=timeout,
            )
            if not dataset_results:
                return

            if stats := dataset_results.runtime_stats():
                ui.writeln()
                _write_stats(stats)

            if sol.is_partial:
                if dataset_results.has_validation_errors():
                    ui.writeln()
                    ui.writeln(
                        "The results don't match the solution's expected outcome.\n",
                        ui.ERROR,
                    )
                    ui.writeln(
                        "Subtasks with issues:",
                        ui.ERROR,
                    )
                    for sti, err in dataset_results.validation.items():
                        if isinstance(err, Error):
                            ui.writeln(f" * {sti!r}: {err.msg}", ui.ERROR)
                else:
                    ui.writeln()
                    ui.writeln(
                        "Solution produced the expected results",
                        ui.OK,
                    )
            else:
                ui.writeln()
                if dataset_results.check_all_correct():
                    ui.writeln("Result: All test passed", ui.OK)
                else:
                    ui.writeln("Result: Some tests failed", ui.ERROR)

    @ui.hd1("{0}", "Checking dataset", COLOR)
    def check_dataset(self) -> Status:
        """Check input/output correctness.

        First run all correct solutions against all test cases and sample input. Then use the running
        time of correct solutions to set a timeout. Finally, use the timeout to run partial solutions
        and ensure they fail the subtasks they are suppose to fail.
        """
        if sum(c for c in self._dataset.counts().values()) == 0:
            ui.show_message(
                "Error",
                "No test cases found. Generate the dataset by running `ocimatic run-testplan && ocimatic gen-expected`.",
                ui.ERROR,
            )
            return Status.fail

        if not self._dataset.check_all_have_expected():
            ui.show_message(
                "Error",
                "Some test cases don't have expected output, generate them with `ocimatic gen-expected`.",
                ui.ERROR,
            )
            return Status.fail

        # Do not early return if there are input/output validation errors but still report them at the end
        validate_input_status = self._check_dataset_validate_input()
        validate_output_status = self._check_dataset_validate_output()

        timeout = self._check_dataset_run_correct_solutions()
        if not timeout:
            return Status.fail

        return (
            self._check_dataset_run_partial_solutions(timeout)
            & validate_input_status
            & validate_output_status
        )

    def _check_dataset_validate_input(self) -> Status:
        ui.writeln()
        ui.writeln("Validating input files", ui.INFO)
        ui.writeln()
        status = self._dataset.validate_input(stn=None)
        ui.writeln()
        if status == Status.fail:
            ui.writeln("Some subtasks didn't pass input validation.", ui.ERROR)
            ui.writeln()

        return status

    def _check_dataset_validate_output(self) -> Status:
        ui.writeln()
        ui.writeln("Validating expected output files", ui.INFO)
        ui.writeln()
        status = self._dataset.validate_output(stn=None)
        ui.writeln()
        if status == Status.fail:
            ui.writeln("Some subtasks didn't pass output validation.", ui.ERROR)
            ui.writeln()

        return status

    def _check_dataset_run_correct_solutions(self) -> float | None:
        included = [sol for sol in self._correct if sol.should_include_in_stats()]
        excluded = [sol for sol in self._correct if not sol.should_include_in_stats()]

        if not included:
            ui.show_message(
                "Error",
                "at least one correct solution must be included in the runtime stats",
                ui.ERROR,
            )
            return None

        failed: list[Solution] = []

        ui.writeln("Running correct solutions included in stats", ui.INFO)
        stats = RuntimeStats.unit()
        for sol in included:
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

        ui.writeln()
        _write_stats(stats)
        ui.writeln()
        timeout = stats.set_limit()
        ui.writeln(
            f"Timeout set to {timeout:.1f}s ({stats.fmt_limit_calculation()})",
        )

        if excluded:
            ui.writeln()
            ui.writeln("Running correct solutions excluded from stats", ui.INFO)
            for sol in excluded:
                results = sol.run_on_dataset(
                    self._dataset,
                    self._checker,
                    RunMode.check_correct,
                    timeout=timeout,
                )
                if results is None or not results.check_all_correct():
                    failed.append(sol)

        if failed:
            ui.write(
                """
Summary
-------
Some correct solutions failed to run or produced wrong results. Run them individually with
`ocimatic run` to get more information.

Solutions with issues:
""",
                ui.RED,
            )

            for sol in failed:
                ui.writeln(" * " + str(sol), ui.RED)
            return None

        ui.writeln()
        ui.writeln("All correct solutions produced correct results", ui.GREEN)
        return timeout

    def _check_dataset_run_partial_solutions(self, timeout: float) -> Status:
        ui.writeln()
        ui.writeln("Running partial solutions", ui.INFO)
        if not self._partial:
            ui.writeln()
            ui.writeln("warning: no partial solutions", ui.YELLOW)
            return Status.success

        failed: list[Solution] = []
        for sol in self._partial:
            results = sol.run_on_dataset(
                self._dataset,
                self._checker,
                RunMode.check_partial,
                timeout=timeout,
            )
            if results is None or results.has_validation_errors():
                if len(self._partial) > 1:
                    ui.writeln("error: issues found", ui.RED)
                failed.append(sol)

        if failed:
            ui.write(
                """
Summary
-------
Some partial solutions had issues when running or didn't pass/fail the subtasks they were supposed to.
Run them individually with `ocimatic run` to get more information. When running a solution individually,
remember to set an appropriate timeout using the `--timeout` flag.

Solutions with issues:
""",
                ui.RED,
            )
            for sol in failed:
                ui.writeln(" * " + str(sol), ui.RED)
            return Status.fail

        ui.writeln()
        ui.writeln(
            "All partial solutions passed/failed the subtasks they were supposed to.",
            ui.GREEN,
        )
        return Status.success

    @ui.hd1("{0}", "Building solutions", COLOR)
    def build_solution(self, solution: Path) -> None:
        """Force compilation of solutions."""
        sol = self.load_solution_from_path(solution)
        if not sol:
            return ui.show_message("Error", "Solution not found", ui.ERROR)
        sol.build()

    @ui.hd1("{0}", "Generating expected output", COLOR)
    def gen_expected(
        self,
        *,
        sample: bool = False,
        solution: Path | None = None,
    ) -> Status:
        """Generate expected outputs files for the dataset by running a correct solution.

        If `sample` is True, also generate expected output for sample input. If `solution` is
        not `None` use it to generate the expected output, otherwise use any correct one,
        prioritizing C++ solutions.
        """
        if self._conf.dataset.static:
            ui.show_message("skipping", "task has a static dataset.")
            return Status.success
        if not self._correct:
            ui.show_message("error", "no correct solution.", ui.RED)
            return Status.fail
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
            ui.fatal_error("solution not found")
        if generator.gen_expected(self._dataset, sample=sample) == Status.fail:
            return Status.fail

        if sum(c for c in self._dataset.counts().values()) == 0:
            ui.show_message("warning", "empty dataset", ui.WARNING)

        return Status.success

    @ui.hd1("{0}", "Building statement", COLOR)
    def build_statement(self) -> None:
        """Generate pdf for the statement."""
        self._statement.build()

    @ui.hd1("{0}", "Sync", COLOR)
    def sync_resources(self, typesetting: Typesetting) -> None:
        statement_dir = self._directory / "statement"
        for resource in Task.STATEMENT_RESOURCES[typesetting]:
            resource.sync(statement_dir)


class Statement(ABC):
    @staticmethod
    def load(
        directory: Path,
        typesetting: Typesetting,
        *,
        phase: str,
        num: int,
        codename: str,
    ) -> Statement:
        match typesetting:
            case Typesetting.LATEX:
                return LatexStatement(
                    directory,
                    phase=phase,
                    num=num,
                    codename=codename,
                )
            case Typesetting.TYPST:
                return TypstStatement(
                    directory,
                    phase=phase,
                    num=num,
                    codename=codename,
                )

    def __init__(
        self,
        directory: Path,
        source: PDFSource,
        num: int,
        codename: str,
    ) -> None:
        self._directory = directory
        self._source = source
        self._num = num
        self._codename = codename

    @abstractmethod
    def _get_title_from_source(self) -> str | None: ...

    @abstractmethod
    def _get_io_samples_from_source(self) -> set[str]: ...

    @abstractmethod
    def _get_scores_from_source(self) -> SortedDict[Stn, int]: ...

    def get_io_samples(self) -> list[Test]:
        """Find sample input data in the statement."""
        return [
            Test(self._directory / f"{s}.in", self._directory / f"{s}.sol")
            for s in self._get_io_samples_from_source()
        ]

    def get_title(self) -> str:
        title = self._get_title_from_source() or self._codename or self._directory.name
        return f"Problema {_number_to_letter(self._num)} - {title}"

    def get_scores(self) -> SortedDict[Stn, int]:
        """Find the scores for each subtask."""
        scores: SortedDict[Stn, int] = self._get_scores_from_source()
        if not scores:
            ui.show_message(
                "warning",
                "couldn't infer the score from the statement, assuming a single subtask with 100 points.",
                ui.WARNING,
            )
            scores[Stn(1)] = 100

        return scores

    def build(self) -> Result:
        return self._source.compile_work()

    def __str__(self) -> str:
        return str(self._source)

    @property
    def pdf(self) -> Path | None:
        """Return path to pdf if exists."""
        return self._source.pdf()


class TypstStatement(Statement):
    _TITLE_RE = re.compile(r"\s*title:\s*\"([^\"]+)\"")
    _SAMPLE_IO_RE: re.Pattern[str] = re.compile(r"\s*#sampleIO\(\"([^\"]+)\"\)")
    _SUBTASK_RE = re.compile(r"\s*#subtask\(([^\)]+)\)")

    def __init__(
        self,
        directory: Path,
        *,
        phase: str,
        num: int,
        codename: str,
    ) -> None:
        statement_path = directory / "statement.typ"
        assert statement_path.exists(), f"{statement_path} does not exist"
        sys_inputs: dict[str, str] = {
            "OCIMATIC_PHASE": phase,
            "OCIMATIC_PROBLEM_NUMBER": _number_to_letter(num),
            "OCIMATIC_CODENAME": codename,
        }
        source = TypstSource(statement_path, sys_inputs=sys_inputs)
        super().__init__(directory, source, num, codename)

    def _get_title_from_source(self) -> str | None:
        m = next((_match_lines(self._source.iter_lines(), self._TITLE_RE)), None)
        return m and m.group(1)

    def _get_io_samples_from_source(self) -> set[str]:
        return {
            m.group(1)
            for m in _match_lines(self._source.iter_lines(), self._SAMPLE_IO_RE)
        }

    def _get_scores_from_source(self) -> SortedDict[Stn, int]:
        """Find the scores for each subtask."""
        scores: SortedDict[Stn, int] = SortedDict()
        sti = 1
        for m in _match_lines(self._source.iter_lines(), self._SUBTASK_RE):
            scores[Stn(sti)] = int(m.group(1))
            sti += 1
        return scores


class LatexStatement(Statement):
    """Represents a statement. A statement is composed by a latex source and a pdf file."""

    _SAMPLE_IO_RE = re.compile(r"[^%]*\\sampleIO(?:\*)?(\[[^\]]*\]){0,2}{([^}]+)}")
    _SUBTASK_RE = re.compile(r"[^%]*\\subtask{([^}]+)}")
    _TITLE_RE = re.compile(r"\\title{([^}]+)}")

    def __init__(
        self,
        directory: Path,
        *,
        phase: str,
        num: int,
        codename: str,
    ) -> None:
        assert (directory / "statement.tex").exists()
        env: dict[str, str] = {
            "OCIMATIC_PHASE": phase,
            "OCIMATIC_PROBLEM_NUMBER": _number_to_letter(num),
            "OCIMATIC_CODENAME": codename,
        }
        source = LatexSource(directory / "statement.tex", env=env)
        super().__init__(directory, source, num, codename)

    def _get_title_from_source(self) -> str | None:
        m = next((_match_lines(self._source.iter_lines(), self._TITLE_RE)), None)
        return m and m.group(1)

    def _get_io_samples_from_source(self) -> set[str]:
        return {
            m.group(2)
            for m in _match_lines(self._source.iter_lines(), self._SAMPLE_IO_RE)
        }

    def _get_scores_from_source(self) -> SortedDict[Stn, int]:
        scores: SortedDict[Stn, int] = SortedDict()
        sti = 1
        for m in _match_lines(self._source.iter_lines(), self._SUBTASK_RE):
            scores[Stn(sti)] = int(m.group(1))
            sti += 1
        return scores


def _match_lines(
    lines: Iterable[str],
    pattern: re.Pattern[str],
) -> Iterator[re.Match[str]]:
    for line in lines:
        m = pattern.match(line)
        if m:
            yield m


def _number_to_letter(num: int) -> str:
    return chr(ord("A") + num)


def _write_stats(stats: RuntimeStats) -> None:
    ui.writeln("Running time")
    ui.writeln(f"  Max: {stats.max:.3f}s")
    ui.writeln(f"  Min: {stats.min:.3f}s")


class Sideness(Enum):
    ONESIDE = 0
    TWOSIDE = 1

    def __str__(self) -> str:
        match self:
            case Sideness.ONESIDE:
                return "oneside"
            case Sideness.TWOSIDE:
                return "twoside"


class Evenness(Enum):
    EVEN = 0
    ODD = 1

    def check(self, n: int) -> bool:
        return n % 2 == self.value
