from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
import re

from ocimatic import ui, utils
from ocimatic.checkers import Checker
from ocimatic.dataset import (
    Dataset,
    Outcome,
    DatasetResults,
    GroupResults,
    RunMode,
)
from ocimatic.result import Result, Status, Error
from ocimatic.source_code import (
    BuildError,
    CppSource,
    JavaSource,
    PythonSource,
    RustSource,
    SourceCode,
)
from ocimatic.utils import SortedDict, Stn


class Solution:
    VALID_EXTENSIONS = (".cpp", ".java", ".py", ".rs")

    COLOR = ui.BLUE

    def __init__(self, source: SourceCode) -> None:
        self._source = source

        self.is_partial = source.file.parent.name == "partial"

    @staticmethod
    def load_solutions_in_dir(
        codename: str,
        directory: Path,
        managers_dir: Path,
    ) -> list[Solution]:
        """Search for solutions in a directory."""
        assert directory.is_dir()
        return [
            solution
            for file_path in directory.iterdir()
            for solution in [Solution.load(codename, file_path, managers_dir)]
            if solution
        ]

    @staticmethod
    def load(
        codename: str,
        source_path: Path,
        managers_dir: Path,
    ) -> Solution | None:
        if not source_path.is_file():
            return None

        source: SourceCode
        if source_path.suffix == CppSource.SUFFIX:
            grader = Path(managers_dir, "grader.cpp")
            source = CppSource(
                source_path,
                extra_files=[grader] if grader.exists() else [],
                include=managers_dir,
            )
        elif source_path.suffix == JavaSource.SUFFIX:
            source = JavaSource(codename, source_path)
        elif source_path.suffix == PythonSource.SUFFIX:
            source = PythonSource(source_path)
        elif source_path.suffix == RustSource.SUFFIX:
            source = RustSource(source_path)
        else:
            return None

        return Solution(source)

    @ui.workhd("{0}", COLOR)
    def run_on_subtask(
        self,
        dataset: Dataset,
        checker: Checker,
        stn: Stn,
        *,
        timeout: float | None = None,
    ) -> ui.WorkHd[GroupResults | None]:
        """Run this solution on one subtask."""
        build_result = self._source.build()
        if isinstance(build_result, BuildError):
            yield Result.fail(short_msg="Failed", long_msg=build_result.msg)
            return None
        yield Result.success(short_msg="OK")
        return dataset.subtask(stn).run_on(
            build_result,
            checker,
            RunMode.run_solution,
            timeout=timeout,
        )

    def load_expected_outcome(
        self,
        dataset: Dataset,
    ) -> SortedDict[Stn, Outcome] | Error:
        if not self.is_partial:
            return SortedDict((sti, Outcome.OK) for sti in dataset.subtasks())

        if isinstance(comments := _parse_comments(self.source), Error):
            return comments
        match len(comments):
            case 0:
                return Error("Missing `@ocimatic::expected` comment")
            case 1:
                expected = comments[0].subtasks
                sts = set(expected.keys())
                if sts != dataset.subtasks():
                    return Error(
                        f"Subtasks in comment don't match dataset\nexpected '{dataset.subtasks()}', got '{sts}'",
                    )
                return expected
            case _:
                return Error(
                    "The `@ocimatic::expected` comment can only be specified once",
                )

    @ui.workhd("{0}", COLOR)
    def run_on_dataset(
        self,
        dataset: Dataset,
        checker: Checker,
        mode: RunMode,
        *,
        timeout: float | None = None,
    ) -> ui.WorkHd[DatasetResults | None]:
        """Run this solution for all test cases in the given dataset."""
        if isinstance(expected := self.load_expected_outcome(dataset), Error):
            yield Result.fail(short_msg="Failed", long_msg=expected.msg)
            return None

        if isinstance(runnable := self._source.build(), BuildError):
            yield Result.fail(short_msg="Failed", long_msg=runnable.msg)
            return None

        yield Result.success(short_msg="OK")
        return dataset.run_on(
            runnable,
            checker,
            mode,
            expected,
            timeout=timeout,
        )

    @ui.workhd("{0}", COLOR)
    def run_on_input(self, input: Path | TextIO) -> ui.WorkHd[None]:
        build_result = self._source.build()
        if isinstance(build_result, BuildError):
            yield Result.fail(short_msg="Failed", long_msg=build_result.msg)
            return None
        else:
            yield Result.success(short_msg="OK")
        build_result.run_on_input(input)

    @ui.workhd("{0}", COLOR)
    def gen_expected(
        self,
        dataset: Dataset,
        *,
        sample: bool = False,
    ) -> ui.WorkHd[Status]:
        """Generate expected output files for all test cases in the given dataset running this solution."""
        build_result = self._source.build()
        if isinstance(build_result, BuildError):
            yield Result.fail(short_msg="Failed", long_msg=build_result.msg)
            return Status.fail
        else:
            yield Result.success(short_msg="OK")
            return dataset.gen_expected(build_result, sample=sample)

    @ui.work("Build")
    def build(self) -> Result:
        """Build solution."""
        result = self._source.build(force=True)
        if isinstance(result, BuildError):
            return Result.fail(short_msg="Failed", long_msg=result.msg)
        else:
            return Result.success(short_msg="OK")

    @property
    def name(self) -> str:
        return self._source.name

    def __str__(self) -> str:
        return str(self.name)

    @property
    def source(self) -> SourceCode:
        return self._source


def _parse_comments(source: SourceCode) -> list[ExpectedComment] | Error:
    pattern = re.compile(r"\s*@ocimatic::expected\s+(.*)")
    comments: list[ExpectedComment] = []
    for comment in source.comments_iter():
        m = pattern.match(comment)
        if not m:
            continue
        parsed = ExpectedComment.parse(m.group(1))
        if isinstance(parsed, Error):
            path = utils.relative_to_cwd(source.file)
            return Error(
                f"Invalid comment `{m.group(0)}` in {path}\n" + parsed.msg,
            )
        else:
            comments.append(parsed)
    return comments


@dataclass(frozen=True, kw_only=True, slots=True)
class ExpectedComment:
    ITEM_RE = re.compile(r"\s*st(\d+)\s*=\s*(\w+)\s*")

    subtasks: SortedDict[Stn, Outcome]

    @staticmethod
    def parse(s: str) -> ExpectedComment | Error:
        s = s.strip()
        if not s.startswith("[") or not s.endswith("]"):
            return Error("Content must be delimited by square brackets")
        s = s[1:-1]

        subtasks: SortedDict[Stn, Outcome] = SortedDict()
        for item in s.split(","):
            m = ExpectedComment.ITEM_RE.match(item)
            if not m:
                return Error(
                    f"Items must be specified in the format `st{{n}}=VAL`, got `{item.strip()}`",
                )
            n = int(m.group(1))
            if n < 1:
                return Error(
                    f"Subtask number must be greater than or equal to 1: `st{n}`",
                )
            stn = Stn(n)
            verdict = Outcome.parse(m.group(2))
            if isinstance(verdict, Error):
                return verdict
            if stn in subtasks:
                return Error("A subtask must appear only once")
            subtasks[stn] = verdict
        return ExpectedComment(subtasks=subtasks)
