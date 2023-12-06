from __future__ import annotations

from pathlib import Path
from typing import TextIO

from ocimatic import utils
from ocimatic.checkers import Checker
from ocimatic.dataset import Dataset, DatasetResults, RunMode
from ocimatic.source_code import (
    BuildError,
    CppSource,
    JavaSource,
    PythonSource,
    RustSource,
    ShouldFail,
    SourceCode,
)
from ocimatic.utils import Stn


class SolutionSpec:
    """Specification of which subtasks should pass or fail."""

    subtasks_spec: ShouldFail | None

    def __init__(self, name: str, comments: list[ShouldFail]) -> None:
        match len(comments):
            case 0:
                self.subtasks_spec = None
            case 1:
                self.subtasks_spec = comments[0]
            case _:
                utils.fatal_error(
                    "The should-fail comment can only be specified once",
                )

    def should_fail(self, data: Dataset) -> set[Stn]:
        """Return the set of subtasks the solution should fail based on the spec. It must pass the complement."""
        all_subtasks = data.subtasks()
        match self.subtasks_spec:
            case ShouldFail(subtasks=subtasks):
                return all_subtasks.intersection(subtasks)
            case None:
                return all_subtasks


class Solution:
    VALID_EXTENSIONS = (".cpp", ".java", ".py", ".rs")

    COLOR = utils.BLUE

    def __init__(self, source: SourceCode) -> None:
        self._source = source
        self._spec = SolutionSpec(source.name, source.comments)
        self.is_partial = source.file.parent.name == "partial"

    def should_fail(self, data: Dataset) -> set[Stn]:
        return self._spec.should_fail(data)

    def check_results(self, results: DatasetResults) -> bool:
        assert self.is_partial
        return results.check_passes_correct_subtasks(self.should_fail(results.dataset))

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

    @utils.workhd("{0}", COLOR)
    def run_on_dataset(
        self,
        dataset: Dataset,
        checker: Checker,
        mode: RunMode,
        *,
        timeout: float | None = None,
        stn: Stn | None = None,
    ) -> utils.WorkHd[DatasetResults | None]:
        """Run this solution for all test cases in the given dataset."""
        build_result = self._source.build()
        if isinstance(build_result, BuildError):
            yield utils.Result.fail(short_msg="Failed", long_msg=build_result.msg)
            return None
        else:
            yield utils.Result.success(short_msg="OK")
            return dataset.run_on(
                build_result,
                checker,
                mode,
                timeout=timeout,
                stn=stn,
            )

    @utils.workhd("{0}", COLOR)
    def run_on_input(self, input: Path | TextIO) -> utils.WorkHd[None]:
        build_result = self._source.build()
        if isinstance(build_result, BuildError):
            yield utils.Result.fail(short_msg="Failed", long_msg=build_result.msg)
            return None
        else:
            yield utils.Result.success(short_msg="OK")
        build_result.run_on_input(input)

    @utils.workhd("{0}", COLOR)
    def gen_expected(
        self,
        dataset: Dataset,
        *,
        sample: bool = False,
    ) -> utils.WorkHd[utils.Status]:
        """Generate expected output files for all test cases in the given dataset running this solution."""
        build_result = self._source.build()
        if isinstance(build_result, BuildError):
            yield utils.Result.fail(short_msg="Failed", long_msg=build_result.msg)
            return utils.Status.fail
        else:
            yield utils.Result.success(short_msg="OK")
            return dataset.gen_expected(build_result, sample=sample)

    @utils.work("Build")
    def build(self) -> utils.WorkResult:
        """Build solution."""
        result = self._source.build(force=True)
        if isinstance(result, BuildError):
            return utils.WorkResult.fail(short_msg="Failed", long_msg=result.msg)
        else:
            return utils.WorkResult.success(short_msg="OK")

    @property
    def name(self) -> str:
        return self._source.name

    def __str__(self) -> str:
        return str(self.name)

    @property
    def source(self) -> SourceCode:
        return self._source
