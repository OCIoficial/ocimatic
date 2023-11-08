from __future__ import annotations

import itertools
import math
import random
import shutil
import string
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import ocimatic
from ocimatic import ui
from ocimatic.checkers import Checker, CheckerError, CheckerSuccess
from ocimatic.runnable import RunError, Runnable, RunSuccess, RunTLE
from ocimatic.source_code import BuildError, CppSource, PythonSource, SourceCode
from ocimatic.ui import WorkResult

IN = ".in"
SOL = ".sol"


class RunMode(Enum):
    run_solution = "run_solution"
    check_correct = "check_correct"
    check_partial = "check_partial"


@dataclass(frozen=True, kw_only=True, slots=True)
class TestResult:
    @dataclass
    class CheckerRunned:
        """The solution run without runtime errors and the checker was succesfully run on the output.

        This could mean a correct answer, a wrong answer, or partial score if the checker returns
        something greater than 0.0 but less than 1.0.
        """

        checker_result: CheckerSuccess
        run_result: RunSuccess

        def running_time(self) -> float:
            return self.run_result.time

        @property
        def outcome(self) -> float:
            return self.checker_result.outcome

        def is_correct(self) -> bool:
            return self.outcome == 1.0

        def into_work_result(self, mode: RunMode) -> WorkResult:
            msg = f"{self.outcome} [{self.run_result.time:.3f}s]"
            if self.checker_result.msg is not None:
                msg += " - %s" % self.checker_result.msg
            if mode is RunMode.check_partial:
                return WorkResult.info(short_msg=msg)
            else:
                return WorkResult(
                    status=ui.Status.from_bool(self.is_correct()),
                    short_msg=msg,
                )

    @dataclass
    class TimeLimitExceeded:
        """The solution exceeded the time limit."""

        run_result: RunTLE

        def into_work_result(self, mode: RunMode) -> WorkResult:
            if mode is RunMode.check_partial:
                return WorkResult.info(short_msg="Execution timed out")
            else:
                return WorkResult.fail(short_msg="Execution timed out")

    @dataclass
    class RuntimeError:
        """The solution had a runtime error."""

        run_result: RunError

        def into_work_result(self, mode: RunMode) -> WorkResult:
            if mode is RunMode.check_partial:
                return WorkResult.info(
                    short_msg=self.run_result.msg,
                    long_msg=self.run_result.stderr,
                )
            else:
                return WorkResult.fail(
                    short_msg=self.run_result.msg,
                    long_msg=self.run_result.stderr,
                )

    @dataclass
    class CheckerError:
        """There was an error running the checker. This means the checker must be fixed."""

        checker_result: CheckerError

        def into_work_result(self, mode: RunMode) -> WorkResult:
            del mode
            msg = f"Failed to run checker: `{self.checker_result.msg}`"
            return WorkResult(status=ui.Status.fail, short_msg=msg)

    @dataclass
    class NoExpectedOutput:
        """The test didn't have a corresponding expectd output.

        This means `ocimatic gen-expected` hasn't been run.
        """

        def into_work_result(self, mode: RunMode) -> WorkResult:
            del mode
            return WorkResult(
                status=ui.Status.fail,
                short_msg="No expected output file",
            )

    def into_work_result(self) -> WorkResult:
        return self.kind.into_work_result(self.mode)

    def is_correct(self) -> bool:
        return (
            isinstance(self.kind, TestResult.CheckerRunned) and self.kind.is_correct()
        )

    def is_proper_fail(self) -> bool:
        """Check whether the test was a "proper" failure.

        A proper failure means the solution itself was wrong (incorrect output, runtime error, or
        time limit exceeded), instead of a failure due to a missing has expected output or an error
        when running the checker.
        """
        match self.kind:
            case TestResult.RuntimeError(_) | TestResult.TimeLimitExceeded(_):
                return True
            case TestResult.CheckerRunned(_):
                return not self.kind.is_correct()
            case _:
                return False

    Kind = (
        CheckerRunned
        | TimeLimitExceeded
        | RuntimeError
        | CheckerError
        | NoExpectedOutput
    )
    kind: Kind
    mode: RunMode


class Test:
    """A single test file.

    A test is composed of an input and an expected output. The input file must be present in the
    file system, but the expected output may be missing.
    """

    def __init__(self, in_path: Path, expected_path: Path) -> None:
        assert in_path.exists()
        self._in_path = in_path
        self._expected_path = expected_path

    def __str__(self) -> str:
        return str(self._in_path.relative_to(ocimatic.config["contest_root"]))

    def mtime(self) -> float:
        if self._expected_path.exists():
            return max(
                self._in_path.stat().st_mtime,
                self._expected_path.stat().st_mtime,
            )
        return self._in_path.stat().st_mtime

    @ui.work("Validate", "{0}")
    def validate(self, validator: Runnable) -> WorkResult:
        result = validator.run(in_path=self._in_path)
        match result:
            case RunSuccess(_):
                return WorkResult.success(short_msg="OK")
            case RunError(msg, stderr):
                return WorkResult.fail(short_msg=msg, long_msg=stderr)

    @ui.work("Gen")
    def gen_expected(self, runnable: Runnable) -> WorkResult:
        """Run binary with this test as input to generate expected output file."""
        result = runnable.run(in_path=self.in_path, out_path=self.expected_path)
        match result:
            case RunSuccess(_):
                return WorkResult.success(short_msg="OK")
            case RunError(msg, stderr):
                return WorkResult.fail(short_msg=msg, long_msg=stderr)

    @ui.work("Run")
    def run(
        self,
        runnable: Runnable,
        checker: Checker,
        mode: RunMode,
        timeout: float | None,
    ) -> TestResult:
        """Run runnable redirecting this test as its standard input and check output correctness."""
        kind = self.run_inner(runnable, checker, timeout)
        return TestResult(mode=mode, kind=kind)

    def run_inner(
        self,
        runnable: Runnable,
        checker: Checker,
        timeout: float | None,
    ) -> TestResult.Kind:
        if not self.expected_path.exists():
            return TestResult.NoExpectedOutput()

        # We could use NamedTemporaryFile but the documentation says not every platform can use the
        # name to reopen the file while still open, so we create a temporary directory and a file
        # inside it instead.
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir, "out")

            run_result = runnable.run(
                in_path=self.in_path,
                out_path=out_path,
                timeout=timeout,
            )

            if isinstance(run_result, RunError):
                return TestResult.RuntimeError(run_result)

            if isinstance(run_result, RunTLE):
                return TestResult.TimeLimitExceeded(run_result)

            checker_result = checker.run(
                in_path=self.in_path,
                expected_path=self.expected_path,
                out_path=out_path,
            )

            # Checker failed
            if isinstance(checker_result, CheckerError):
                return TestResult.CheckerError(checker_result)

            return TestResult.CheckerRunned(checker_result, run_result)

    @property
    def in_path(self) -> Path:
        return self._in_path

    @property
    def expected_path(self) -> Path:
        return self._expected_path

    @ui.work("Normalize")
    def normalize(self) -> WorkResult:
        if not shutil.which("dos2unix"):
            return WorkResult.fail(short_msg="Cannot find dos2unix")
        if not shutil.which("sed"):
            return WorkResult.fail(short_msg="Cannot find sed")
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
        return WorkResult(
            status=ui.Status.from_bool(st == 0),
            short_msg="OK" if st == 0 else "FAILED",
        )

    def has_expected(self) -> bool:
        return self._expected_path.exists()


class TestGroup:
    """A collection of test cases."""

    def __init__(self, name: str, tests: list[Test]) -> None:
        self._name = name
        self._tests = tests

    def write_to_zip(self, zip_file: ZipFile, *, random_sort: bool = False) -> int:
        copied = 0
        for test in self._tests:
            if test.expected_path.exists():
                # Sort testcases within a subtask randomly
                if random_sort:
                    choices = string.ascii_lowercase
                    rnd_str = "".join(random.choice(choices) for _ in range(3))
                    in_name = f"{self._name}-{rnd_str}-{test.in_path.name}"
                    sol_name = f"{self._name}-{rnd_str}-{test.expected_path.name}"
                else:
                    in_name = f"{self._name}-{test.in_path.name}"
                    sol_name = f"{self._name}-{test.expected_path.name}"
                zip_file.write(test.in_path, in_name)
                zip_file.write(test.expected_path, sol_name)
                copied += 1
        return copied

    def mtime(self) -> float:
        mtime = -1.0
        for test in self._tests:
            mtime = max(mtime, test.mtime())
        return mtime

    def normalize(self) -> None:
        for test in self._tests:
            test.normalize()

    def count(self) -> int:
        return len(self._tests)

    @ui.workgroup("{0}")
    def run(
        self,
        runnable: Runnable,
        checker: Checker,
        mode: RunMode,
        *,
        timeout: float | None,
        skip: bool = False,
    ) -> list[TestResult] | None:
        if skip:
            ui.writeln(" Info: Skipping")
            return None
        results: list[TestResult] = []
        for test in self._tests:
            result = test.run(runnable, checker, mode, timeout)
            results.append(result)
        return results

    @ui.workgroup("{0}")
    def gen_expected(
        self,
        runnable: Runnable,
    ) -> Literal[ui.Status.success, ui.Status.fail]:
        status = ui.Status.success
        for test in self._tests:
            if test.gen_expected(runnable).status is not ui.Status.success:
                status = ui.Status.fail
        return status

    def tests(self) -> Iterable[Test]:
        return self._tests

    def __str__(self) -> str:
        return self._name


class Subtask(TestGroup):
    """Subclass of `TestGroup` to represet a subtask."""

    def __init__(self, directory: Path) -> None:
        super().__init__(
            directory.name,
            [Test(f, f.with_suffix(SOL)) for f in directory.glob(f"*{IN}")],
        )

    @ui.workgroup("{0}")
    def validate(self, validator: Path | None) -> None:
        if validator is None:
            ui.show_message("Info", "No validator specified", ui.INFO)
            return
        source: SourceCode
        if validator.suffix == ".cpp":
            source = CppSource(validator)
        elif validator.suffix == ".py":
            source = PythonSource(validator)
        else:
            ui.show_message("Warning", "Unsupported file for validator", ui.WARNING)
            return
        build = source.build()
        if isinstance(build, BuildError):
            ui.show_message(
                "Warning",
                f"Failed to build validator\n{build.msg}",
                ui.WARNING,
            )
            return
        for test in self._tests:
            test.validate(build)


@dataclass
class RuntimeStats:
    max: float
    min: float

    @staticmethod
    def unit() -> RuntimeStats:
        return RuntimeStats(max=float("-inf"), min=float("inf"))

    def set_limit(self) -> float | None:
        return math.ceil(self.max * 20) / 10 if self.max else None

    def print_limit_calculation(self) -> str:
        return f"math.ceil({self.max:.3f} * 20) / 10"

    def __add__(self, other: RuntimeStats) -> RuntimeStats:
        return RuntimeStats(max=max(self.max, other.max), min=min(self.min, other.min))

    def __iadd__(self, other: RuntimeStats) -> RuntimeStats:
        return self + other


@dataclass
class DatasetResults:
    subtasks: list[list[TestResult] | None]
    sample: list[TestResult] | None

    def check_all_correct(self) -> bool:
        """Return whether all test cases have a correct answer."""
        for test in self._iter_all(include_sample=True):
            if not isinstance(test.kind, TestResult.CheckerRunned):
                return False
            if not test.kind.is_correct():
                return False
        return True

    def check_passes_correct_subtasks(self, should_pass: set[int]) -> bool:
        """Check all subtasks specified in `should_pass` are correct and the rest fail."""
        for st, tests in enumerate(self.subtasks):
            assert tests, f"Subtask {st} has no test results"
            in_should_pass = (st + 1) in should_pass
            if in_should_pass and not all(t.is_correct() for t in tests):
                return False
            if not in_should_pass and not any(t.is_proper_fail() for t in tests):
                return False

        return True

    def runtime_stats(self, *, include_sample: bool = False) -> RuntimeStats | None:
        running_times = list(self._runnint_times(include_sample=include_sample))
        if not running_times:
            return None
        return RuntimeStats(max=max(running_times), min=min(running_times))

    def _iter_all(self, *, include_sample: bool = False) -> Iterator[TestResult]:
        tests: Iterable[TestResult] = (t for st in self.subtasks if st for t in st)
        if include_sample and self.sample:
            tests = itertools.chain(tests, self.sample)
        yield from tests

    def _runnint_times(self, *, include_sample: bool = False) -> Iterator[float]:
        """Return running times of all successful runs."""
        for test in self._iter_all(include_sample=include_sample):
            if isinstance(test.kind, TestResult.CheckerRunned):
                yield test.kind.running_time()


class Dataset:
    """A collection of test cases."""

    def __init__(self, directory: Path, sampledata: list[Test]) -> None:
        self._directory = directory
        if directory.exists():
            self._subtasks = [
                Subtask(d) for d in sorted(directory.iterdir()) if d.is_dir()
            ]
        else:
            self._subtasks = []
        self._sampledata = TestGroup("sample", sampledata)

    def gen_expected(
        self,
        runnable: Runnable,
        *,
        sample: bool = False,
    ) -> Literal[ui.Status.success, ui.Status.fail]:
        status = ui.Status.success
        for subtask in self._subtasks:
            if subtask.gen_expected(runnable) is not ui.Status.success:
                status = ui.Status.fail
        if sample and self._sampledata.gen_expected(runnable) is not ui.Status.fail:
            status = ui.Status.fail
        return status

    def run(
        self,
        runnable: Runnable,
        checker: Checker,
        mode: RunMode,
        *,
        timeout: float | None = None,
        subtask: int | None = None,
    ) -> DatasetResults:
        subtasks: list[list[TestResult] | None] = []
        for i, st in enumerate(self._subtasks):
            skip = subtask is not None and subtask != i + 1
            subtasks.append(st.run(runnable, checker, mode, timeout=timeout, skip=skip))

        sample = None
        if mode is not RunMode.check_partial:
            sample = self._sampledata.run(
                runnable,
                checker,
                mode,
                timeout=timeout,
                skip=subtask is not None,
            )
        return DatasetResults(subtasks, sample)

    def validate(self, validators: list[Path | None], stn: int | None) -> None:
        zipped = zip(self._subtasks, validators, strict=True)
        for i, (subtask, validator) in enumerate(zipped, 1):
            if stn is None or stn == i:
                subtask.validate(validator)

    def __str__(self) -> str:
        return f"{self._directory}"

    def mtime(self) -> float:
        mtime = -1.0
        for subtask in self._subtasks:
            mtime = max(mtime, subtask.mtime())
        return mtime

    @ui.work("ZIP")
    def compress(self, *, random_sort: bool = False) -> ui.Result:
        """Compress all test cases in the dataset into a single zip file.

        The basename of the corresponding subtask subdirectory is prepended to each file.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        path = Path(self._directory, "data.zip")
        with ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zip:
            compressed = 0
            for subtask in self._subtasks:
                compressed += subtask.write_to_zip(zip, random_sort=random_sort)

            if compressed == 0:
                path.unlink()
                return ui.Result.fail("EMPTY DATASET")
        return ui.Result.success("OK")

    def count(self) -> list[int]:
        return [st.count() for st in self._subtasks]

    def normalize(self) -> None:
        for subtask in self._subtasks:
            subtask.normalize()
        self._sampledata.normalize()

    def check_all_have_expected(self) -> bool:
        for st in self._subtasks:
            for test in st.tests():
                if not test.has_expected():
                    return False
        return True
