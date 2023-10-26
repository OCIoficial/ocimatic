from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ocimatic import ui
from ocimatic.checkers import Checker
from ocimatic.dataset import Dataset, DatasetResults, RunMode
from ocimatic.source_code import (
    BuildError,
    CppSource,
    JavaSource,
    OcimaticComment,
    PythonSource,
    RustSource,
    ShouldFail,
    ShouldPass,
    SourceCode,
)


@dataclass
class SolutionComment:
    should_fail: set[int]


class SolutionSpec:
    """Specification of which subtasks should pass or fail."""

    subtasks_spec: ShouldFail | ShouldPass | None

    def __init__(self, name: str, comments: list[OcimaticComment]) -> None:
        match len(comments):
            case 0:
                self.subtasks_spec = None
            case 1:
                self.subtasks_spec = comments[0]
            case _:
                ui.fatal_error(
                    f"Only on of should-pass or should-fail should be specified: {name}",
                )

    def should_pass(self, results: DatasetResults) -> set[int]:
        """Return the set of subtasks the solution should pass based on the spec. It must fail the complement."""
        all_subtasks = {st + 1 for st in range(len(results.subtasks))}
        match self.subtasks_spec:
            case ShouldFail(subtasks=subtasks):
                return all_subtasks.difference(subtasks)
            case ShouldPass(subtasks=subtasks):
                return all_subtasks.intersection(subtasks)
            case None:
                return set()


class Solution:
    def __init__(self, source: SourceCode) -> None:
        self._source = source
        self._spec = SolutionSpec(source.name, source.comments)
        self.is_partial = source.file.parent.name == "partial"

    def should_pass(self, results: DatasetResults) -> set[int]:
        return self._spec.should_pass(results)

    def check_results(self, results: DatasetResults) -> bool:
        assert self.is_partial
        return results.check_passes_correct_subtasks(
            should_pass=self.should_pass(results),
        )

    @staticmethod
    def load_solutions_in_dir(
        codename: str,
        directory: Path,
        managers_dir: Path,
    ) -> list["Solution"]:
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
    ) -> Optional["Solution"]:
        if not source_path.is_file():
            return None

        source: SourceCode
        if source_path.suffix == ".cpp":
            grader = Path(managers_dir, "grader.cpp")
            source = CppSource(
                source_path,
                extra_files=[grader] if grader.exists() else [],
                include=managers_dir,
            )
        elif source_path.suffix == ".java":
            source = JavaSource(codename, source_path)
        elif source_path.suffix == ".py":
            source = PythonSource(source_path)
        elif source_path.suffix == ".rs":
            source = RustSource(source_path)
        else:
            return None

        return Solution(source)

    @ui.solution_group()
    def run(
        self,
        dataset: Dataset,
        checker: Checker,
        mode: RunMode,
        timeout: float | None,
    ) -> ui.SolutionGroup[DatasetResults | None]:
        """Run this solution for all test cases in the given dataset."""
        build_result = self._source.build()
        if isinstance(build_result, BuildError):
            yield ui.Result.fail(short_msg="Failed", long_msg=build_result.msg)
            return None
        else:
            yield ui.Result.success(short_msg="OK")
            return dataset.run(build_result, checker, mode, timeout)

    @ui.solution_group()
    def gen_expected(
        self,
        dataset: Dataset,
        *,
        sample: bool = False,
    ) -> ui.SolutionGroup[None]:
        """Generate expected output files for all test cases in the given dataset running this solution."""
        build_result = self._source.build()
        if isinstance(build_result, BuildError):
            yield ui.Result.fail(short_msg="Failed", long_msg=build_result.msg)
        else:
            yield ui.Result.success(short_msg="OK")
            dataset.gen_expected(build_result, sample=sample)

    @ui.work("Build")
    def build(self) -> ui.WorkResult:
        """Build solution."""
        result = self._source.build(force=True)
        if isinstance(result, BuildError):
            return ui.WorkResult.fail(short_msg="Failed", long_msg=result.msg)
        else:
            return ui.WorkResult.success(short_msg="OK")

    @property
    def name(self) -> str:
        return self._source.name

    def __str__(self) -> str:
        return str(self.name)

    @property
    def source(self) -> SourceCode:
        return self._source
        return self._source
