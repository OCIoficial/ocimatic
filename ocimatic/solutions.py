from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, TextIO

from ocimatic import ui, utils
from ocimatic.checkers import Checker
from ocimatic.dataset import (
    Dataset,
    DatasetResults,
    GroupResults,
    Outcome,
    RunMode,
)
from ocimatic.result import Error, Result, Status
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
        if isinstance(runnable := self._source.build(), BuildError):
            yield Result.fail(short_msg="Failed", long_msg=runnable.msg)
            return None
        yield Result.success(short_msg="OK")
        return dataset.subtask(stn).run_on(
            runnable,
            checker,
            RunMode.run_solution,
            timeout=timeout,
        )

    def load_expected_outcome(
        self,
        dataset: Dataset,
    ) -> SortedDict[Stn, Outcome] | Error:
        if isinstance(comments := _parse_comments(self.source, ExpectedComment), Error):
            return comments
        match len(comments):
            case 0:
                if self.is_partial:
                    return Error("Missing `@ocimatic::expected` comment")
                else:
                    return SortedDict((sti, Outcome.OK) for sti in dataset.subtasks())
            case 1:
                expected = comments[0].subtasks
                sts = set(expected.keys())
                if sts != dataset.subtasks():
                    return Error(
                        f"Subtasks in comment don't match dataset\nexpected '{dataset.subtasks()}', got '{sts}'",
                    )
                if not self.is_partial and any(
                    out not in [Outcome.OK, Outcome.TLE] for out in expected.values()
                ):
                    return Error(
                        "Correct solutions can only be annotated with 'OK' or 'TLE'",
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
        stn: Stn | None,
        sample: bool = False,
    ) -> ui.WorkHd[Status]:
        """Generate expected output files for all test cases in the given dataset running this solution."""
        build_result = self._source.build()
        if isinstance(build_result, BuildError):
            yield Result.fail(short_msg="Failed", long_msg=build_result.msg)
            return Status.fail
        else:
            yield Result.success(short_msg="OK")
            return dataset.gen_expected(build_result, stn=stn, sample=sample)

    @ui.work("Build")
    def build(self) -> Result:
        """Build solution."""
        result = self._source.build(force=True)
        if isinstance(result, BuildError):
            return Result.fail(short_msg="Failed", long_msg=result.msg)
        else:
            return Result.success(short_msg="OK")

    @ui.workhd("{0}", COLOR)
    def coverage(self, dataset: Dataset) -> ui.WorkHd[None]:
        """Build solution."""
        if not isinstance(self._source, CppSource):
            yield Result.success(short_msg="OK")
            ui.writeln("Coverage analysis is only supported for C++")
            return

        result = self._source.build_for_coverage()
        if isinstance(result, BuildError):
            yield Result.fail(short_msg="Failed", long_msg=result.msg)
        else:
            yield Result.success(short_msg="OK")
            for test in dataset.all_tests(sample=False):
                ui.write(".", flush=True)
                result.run_on_input(
                    test.in_path,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            ui.writeln()
            coverage = self._source.compute_coverage()
            if isinstance(coverage, Error):
                ui.write(coverage.msg, ui.RED)
            else:
                ui.writeln(f"Line Coverage:   {coverage.line:.1f}%")
                ui.writeln(f"Branch Coverage: {coverage.branch:.1f}%")

    @property
    def name(self) -> str:
        return self._source.name

    def __str__(self) -> str:
        return str(self.name)

    @property
    def source(self) -> SourceCode:
        return self._source

    def should_include_in_stats(self) -> bool:
        if isinstance(comments := _parse_comments(self.source, IncludeInStats), Error):
            comments = []
        if len(comments) > 0:
            return comments[0].value
        else:
            return not isinstance(self.source, PythonSource)


def _parse_comments[T: Comment](source: SourceCode, t: type[T]) -> list[T] | Error:
    pattern = re.compile(r"\s*@ocimatic::(\S+)\s+(.*)")

    comments: list[T] = []
    for comment in source.comments_iter():
        if not (m := pattern.match(comment)):
            continue
        if m.group(1) != t.name():
            continue

        if isinstance(parsed := t.parse(m.group(2)), Error):
            path = utils.relative_to_cwd(source.file)
            return Error(
                f"Invalid comment `{m.group(0)}` in {path}\n" + parsed.msg,
            )
        else:
            comments.append(parsed)
    return comments


class Comment(Protocol):
    @classmethod
    def name(cls) -> str: ...

    @classmethod
    def parse(cls, s: str) -> Self | Error: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class IncludeInStats:
    value: bool

    BOOL_RE = re.compile(r"\s*(true|false)\s*")

    @classmethod
    def name(cls) -> str:
        return "include-in-stats"

    @classmethod
    def parse(cls, s: str) -> IncludeInStats | Error:
        if not (m := IncludeInStats.BOOL_RE.match(s)):
            return Error("Expected 'true' or 'false'")

        return IncludeInStats(value=m.group(1) == "true")


@dataclass(frozen=True, kw_only=True, slots=True)
class ExpectedComment:
    """Expected outcome for each subtask."""

    ITEM_RE = re.compile(r"\s*st(\d+)\s*=\s*(\w+)\s*")

    subtasks: SortedDict[Stn, Outcome]

    @classmethod
    def name(cls) -> str:
        return "expected"

    @classmethod
    def parse(cls, s: str) -> ExpectedComment | Error:
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
            outcome = Outcome.parse(m.group(2))
            if isinstance(outcome, Error):
                return outcome
            if stn in subtasks:
                return Error("Each subtask must appear only once")
            subtasks[stn] = outcome
        return ExpectedComment(subtasks=subtasks)
