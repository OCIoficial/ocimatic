from __future__ import annotations

import itertools
import math
import random
import re
import shutil
import string
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from zipfile import ZipFile

from ocimatic import utils
from ocimatic.checkers import Checker, CheckerError, CheckerSuccess
from ocimatic.runnable import RunError, Runnable, RunSuccess, RunTLE
from ocimatic.source_code import BuildError, CppSource, PythonSource, SourceCode
from ocimatic.testplan import Testplan
from ocimatic.utils import SortedDict, Stn, WorkResult

IN = ".in"
SOL = ".sol"


class RunMode(Enum):
    run_solution = "run_solution"
    check_correct = "check_correct"
    check_partial = "check_partial"


@dataclass(frozen=True, kw_only=True, slots=True)
class TestResult:
    @dataclass
    class CheckerFinished:
        """The solution finished without runtime errors and the checker was successfully run on the output.

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
                    status=utils.Status.from_bool(self.is_correct()),
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
            return WorkResult(status=utils.Status.fail, short_msg=msg)

    @dataclass
    class NoExpectedOutput:
        """The test didn't have a corresponding expected output.

        This means `ocimatic gen-expected` hasn't been run.
        """

        def into_work_result(self, mode: RunMode) -> WorkResult:
            del mode
            return WorkResult(
                status=utils.Status.fail,
                short_msg="no expected output file",
            )

    def into_work_result(self) -> WorkResult:
        return self.kind.into_work_result(self.mode)

    def is_correct(self) -> bool:
        return (
            isinstance(self.kind, TestResult.CheckerFinished) and self.kind.is_correct()
        )

    def is_proper_fail(self) -> bool:
        """Check whether the test was a "proper" failure.

        A proper failure means the solution itself was wrong (incorrect output, runtime error, or
        time limit exceeded), instead of a failure due to a missing expected output or an error
        when running the checker.
        """
        match self.kind:
            case TestResult.RuntimeError(_) | TestResult.TimeLimitExceeded(_):
                return True
            case TestResult.CheckerFinished(_):
                return not self.kind.is_correct()
            case _:
                return False

    Kind = (
        CheckerFinished
        | TimeLimitExceeded
        | RuntimeError
        | CheckerError
        | NoExpectedOutput
    )
    test: Test
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
        return str(Path(utils.relative_to_cwd(self._in_path)).with_suffix(""))

    def mtime(self) -> float:
        if self._expected_path.exists():
            return max(
                self._in_path.stat().st_mtime,
                self._expected_path.stat().st_mtime,
            )
        return self._in_path.stat().st_mtime

    @utils.work("validate", "{0}.in")
    def validate_input(
        self,
        validator: Runnable | None,
        *,
        check_basic_format: bool,
    ) -> utils.Result:
        if check_basic_format:
            with self.in_path.open() as f:
                fmt_result = _validate_basic_format(f.readlines())
                if fmt_result.is_fail():
                    return fmt_result

        if validator:
            result = validator.run(in_path=self._in_path)
            match result:
                case RunSuccess(_):
                    return utils.Result.success(short_msg="OK")
                case RunError(msg, stderr):
                    return utils.Result.fail(short_msg=msg, long_msg=stderr)
        else:
            return utils.Result.success(short_msg="OK")

    @utils.work("validate", "{0}.sol")
    def validate_output(self) -> utils.Result:
        if not self.expected_path.exists():
            return utils.Result.fail(short_msg="no expected output file")

        with self.expected_path.open() as f:
            return _validate_basic_format(f.readlines())

    @utils.work("gen")
    def gen_expected(self, runnable: Runnable) -> utils.Result:
        """Run binary with this test as input to generate expected output file."""
        result = runnable.run(in_path=self.in_path, out_path=self.expected_path)
        match result:
            case RunSuccess(_):
                return utils.Result.success(short_msg="OK")
            case RunError(msg, stderr):
                return utils.Result.fail(short_msg=msg, long_msg=stderr)

    @utils.work("run")
    def run_on(
        self,
        runnable: Runnable,
        checker: Checker,
        mode: RunMode,
        timeout: float | None,
    ) -> TestResult:
        """Run runnable redirecting this test as its standard input and check output correctness."""
        kind = self.run_inner(runnable, checker, timeout)
        return TestResult(test=self, mode=mode, kind=kind)

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
        # inside instead.
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

            return TestResult.CheckerFinished(checker_result, run_result)

    @property
    def in_path(self) -> Path:
        return self._in_path

    @property
    def expected_path(self) -> Path:
        return self._expected_path

    @utils.work("Normalize")
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
            status=utils.Status.from_bool(st == 0),
            short_msg="OK" if st == 0 else "FAILED",
        )

    def has_expected(self) -> bool:
        return self._expected_path.exists()

    def __lt__(self, other: Test) -> bool:
        return str(self) < str(other)


class _TestGroup:
    """A collection of test cases."""

    def __init__(self, name: str, tests: list[Test]) -> None:
        self._name = name
        self._tests = sorted(tests)

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

    @utils.hd2("{0}")
    def run_on(
        self,
        runnable: Runnable,
        checker: Checker,
        parents: list[Stn],
        mode: RunMode,
        *,
        timeout: float | None,
        skip: bool = False,
    ) -> list[TestResult] | None:
        if skip:
            utils.show_message("info", "skipping")
            return None

        if mode == RunMode.run_solution:
            for sti in parents:
                utils.writeln(f" @extends st{sti}", utils.CYAN)

        results: list[TestResult] = []
        for test in self._tests:
            result = test.run_on(runnable, checker, mode, timeout)
            results.append(result)
        if not self._tests:
            utils.writeln(" warning: no test cases", utils.YELLOW)

        return results

    @utils.hd2("{0}")
    def gen_expected(self, runnable: Runnable) -> utils.Status:
        status = utils.Status.success
        for test in self._tests:
            status &= test.gen_expected(runnable).status
        return status

    def tests(self) -> Iterable[Test]:
        return self._tests

    def __str__(self) -> str:
        return self._name


class _Subtask(_TestGroup):
    """Subclass of `TestGroup` to represent a subtask."""

    def __init__(self, directory: Path) -> None:
        super().__init__(
            directory.name,
            [Test(f, f.with_suffix(SOL)) for f in directory.glob(f"*{IN}")],
        )

    @utils.hd2("{0}")
    def validate_input(
        self,
        validator: Path | None,
        ancestors: list[_Subtask],
    ) -> utils.Status:
        if validator is None:
            utils.writeln(" warning: no validator available", utils.YELLOW)
            runnable = None
        else:
            runnable_or_error = _build_validator(validator)
            if isinstance(runnable_or_error, utils.Error):
                utils.show_message("error", runnable_or_error.msg, utils.RED)
                return utils.Status.fail
            runnable = runnable_or_error

        status = utils.Status.success

        for st in ancestors:
            for test in st.tests():
                status &= test.validate_input(
                    runnable,
                    check_basic_format=False,
                ).status

        for test in self._tests:
            status &= test.validate_input(runnable, check_basic_format=True).status

        if not self._tests:
            utils.writeln(" warning: no test cases", utils.YELLOW)
            return utils.Status.success

        return status

    @utils.hd2("{0}")
    def validate_output(
        self,
    ) -> utils.Status:
        status = utils.Status.success
        for test in self._tests:
            status &= test.validate_output().status

        if not self._tests:
            utils.writeln(" warning: no test cases", utils.YELLOW)
            return utils.Status.success

        return status

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


@dataclass
class RuntimeStats:
    max: float
    min: float

    @staticmethod
    def unit() -> RuntimeStats:
        return RuntimeStats(max=float("-inf"), min=float("inf"))

    def set_limit(self) -> float:
        return math.ceil(self.max * 4) / 2

    def fmt_limit_calculation(self) -> str:
        return f"math.ceil({self.max:.3f} * 4) / 2"

    def __add__(self, other: RuntimeStats) -> RuntimeStats:
        return RuntimeStats(max=max(self.max, other.max), min=min(self.min, other.min))

    def __iadd__(self, other: RuntimeStats) -> RuntimeStats:
        return self + other


@dataclass(kw_only=True, frozen=True, slots=True)
class DatasetResults:
    dataset: Dataset
    subtasks: SortedDict[Stn, SubtaskResults | None]
    sample: list[TestResult] | None

    def check_all_correct(self) -> bool:
        """Return whether all test cases have a correct answer."""
        for test in self._iter_all(include_sample=True):
            if not isinstance(test.kind, TestResult.CheckerFinished):
                return False
            if not test.kind.is_correct():
                return False
        return True

    def check_passes_correct_subtasks(self, should_fail: set[Stn]) -> bool:
        """Check all subtasks specified in `should_fail` fail and the rest pass."""
        for sti, st in self.subtasks.items():
            assert st is not None, f"Subtask {sti} has no test results"
            in_should_fail = sti in should_fail
            if in_should_fail and not any(t.is_proper_fail() for t in st.results(self)):
                return False
            if not in_should_fail and not all(t.is_correct() for t in st.results(self)):
                return False

        return True

    def failed_subtasks(self) -> set[Stn]:
        failed: set[Stn] = set()
        for sti, st in self.subtasks.items():
            assert st, f"Subtask {sti} has no results"
            if not all(t.is_correct() for t in st.results(self)):
                failed.add(sti)
        return failed

    def runtime_stats(self, *, include_sample: bool = False) -> RuntimeStats | None:
        running_times = list(self._running_times(include_sample=include_sample))
        if not running_times:
            return None
        return RuntimeStats(max=max(running_times), min=min(running_times))

    def _iter_all(self, *, include_sample: bool = False) -> Iterator[TestResult]:
        tests: Iterable[TestResult] = (
            t for st in self.subtasks.values() if st for t in st.results(self)
        )
        if include_sample and self.sample:
            tests = itertools.chain(tests, self.sample)
        yield from tests

    def _running_times(self, *, include_sample: bool = False) -> Iterator[float]:
        """Return running times of all successful runs."""
        for test in self._iter_all(include_sample=include_sample):
            if isinstance(test.kind, TestResult.CheckerFinished):
                yield test.kind.running_time()


@dataclass(kw_only=True, frozen=True, slots=True)
class SubtaskResults:
    stn: Stn
    tests: list[TestResult]

    def results(self, results: DatasetResults) -> Iterator[TestResult]:
        yield from self.tests
        testplan = results.dataset.testplan
        if not testplan:
            return

        for sti in results.dataset.ancestors_of(self.stn):
            st = results.subtasks[sti]
            yield from (st.tests if st else [])


class Dataset:
    def __init__(
        self,
        directory: Path,
        testplan: Testplan | None,
        sampledata: list[Test],
    ) -> None:
        self.testplan = testplan
        self._directory = directory

        self._subtasks: SortedDict[Stn, _Subtask]
        if testplan is not None:
            self._subtasks = SortedDict(
                (Stn(stn), _Subtask(directory / f"st{stn}"))
                for stn in range(1, testplan.subtasks + 1)
            )
        elif directory.exists():
            self._subtasks = SortedDict(
                (Stn(stn), _Subtask(d))
                for stn, d in enumerate(sorted(directory.iterdir()), 1)
                if d.is_dir()
            )
        else:
            self._subtasks = SortedDict()

        self._sampledata = _TestGroup("sample", sampledata)

    def subtasks(self) -> set[Stn]:
        return set(self._subtasks.keys())

    def gen_expected(self, runnable: Runnable, *, sample: bool = False) -> utils.Status:
        status = utils.Status.success
        for subtask in self._subtasks.values():
            status &= subtask.gen_expected(runnable)
        if sample:
            status &= self._sampledata.gen_expected(runnable)
        return status

    def run_on(
        self,
        runnable: Runnable,
        checker: Checker,
        mode: RunMode,
        *,
        timeout: float | None = None,
        stn: Stn | None = None,
    ) -> DatasetResults:
        subtasks: SortedDict[Stn, SubtaskResults | None] = SortedDict()
        for sti, st in self._subtasks.items():
            skip = stn is not None and stn != sti
            tests = st.run_on(
                runnable,
                checker,
                self.parents_of(sti),
                mode,
                timeout=timeout,
                skip=skip,
            )
            subtasks[sti] = (
                SubtaskResults(stn=sti, tests=tests) if tests is not None else None
            )

        sample = None
        if mode is not RunMode.check_partial:
            sample = self._sampledata.run_on(
                runnable,
                checker,
                [],
                mode,
                timeout=timeout,
                skip=stn is not None,
            )
        return DatasetResults(
            dataset=self,
            subtasks=subtasks,
            sample=sample,
        )

    def parents_of(self, stn: Stn) -> list[Stn]:
        testplan = self.testplan
        if not testplan:
            return []

        return testplan.parents_of(stn)

    def ancestors_of(self, stn: Stn) -> list[Stn]:
        testplan = self.testplan
        if not testplan:
            return []

        return testplan.ancestors_of(stn)

    def validate_input(self, stn: Stn | None) -> utils.Status:
        validators: list[Path | None] = [None for _ in self._subtasks]
        if self.testplan is not None:
            validators = self.testplan.validators()

        zipped = zip(self._subtasks.items(), validators, strict=True)
        status = utils.Status.success
        for (sti, subtask), validator in zipped:
            if stn is None or stn == sti:
                ancestors = [self._subtasks[stj] for stj in self.ancestors_of(sti)]
                status &= subtask.validate_input(validator, ancestors)
        return status

    def validate_output(self, stn: Stn | None) -> utils.Status:
        status = utils.Status.success
        for sti, subtask in self._subtasks.items():
            if stn is None or stn == sti:
                status &= subtask.validate_output()
        return status

    def __str__(self) -> str:
        return f"{self._directory}"

    def mtime(self) -> float:
        mtime = -1.0
        for subtask in self._subtasks.values():
            mtime = max(mtime, subtask.mtime())
        return mtime

    @utils.work("ZIP")
    def compress(self, *, random_sort: bool = False) -> utils.Result:
        """Compress all test cases in the dataset into a single zip file.

        The basename of the corresponding subtask subdirectory is prepended to each file.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / "data.zip"
        with ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zip:
            compressed = 0
            for subtask in self._subtasks.values():
                compressed += subtask.write_to_zip(zip, random_sort=random_sort)

            if compressed == 0:
                path.unlink()
                return utils.Result.fail("EMPTY DATASET")
        return utils.Result.success("OK")

    def counts(self) -> SortedDict[Stn, int]:
        """Return the number of test cases in each subtask."""
        return SortedDict((stn, st.count()) for stn, st in self._subtasks.items())

    def regexes(self) -> SortedDict[Stn, str]:
        """Return the regex for each subtask that should be included in the params score in cms."""
        regexes: SortedDict[Stn, str] = SortedDict()
        for sti in self._subtasks:
            stns = [sti]
            stns.extend(self.ancestors_of(sti))
            stns.sort()
            joined = "|".join(f"st{stj}" for stj in stns)
            if len(stns) > 1:
                regexes[sti] = f"({joined}).*"
            else:
                regexes[sti] = f"{joined}.*"
        return regexes

    def normalize(self) -> None:
        for subtask in self._subtasks.values():
            subtask.normalize()
        self._sampledata.normalize()

    def check_all_have_expected(self) -> bool:
        for st in self._subtasks.values():
            for test in st.tests():
                if not test.has_expected():
                    return False
        return True


_WS_RE = re.compile(r"\s{2,}")


def _validate_basic_format(lines: list[str]) -> utils.Result:
    for i, line in enumerate(lines):
        err = _validate_basic_line_format(line)
        if err is not None:
            return utils.Result.fail(
                short_msg=f"error in line {i + 1}",
                long_msg=err.msg,
            )

    return utils.Result.success(short_msg="OK")


def _validate_basic_line_format(line: str) -> utils.Error | None:
    if line[-1] != "\n":
        return utils.Error(r"Line doesn't end with '\n'")
    line = line[:-1]

    if not line:
        return utils.Error("Line cannot be empty")

    for c in line:
        if c in "\t\r\f\v":
            return utils.Error(f"Invalid whitespace character: 0x{ord(c):02X}")

    if not line.isascii():
        return utils.Error("Line must contain only ascii characters")

    if line != line.rstrip():
        return utils.Error("Line cannot have trailing whitespaces")

    if line != line.lstrip():
        return utils.Error("Line cannot have leading whitespaces")

    if _WS_RE.search(line):
        return utils.Error("Line cannot contains contiguous whitespaces")

    return None


def _build_validator(path: Path) -> Runnable | utils.Error:
    source = _load_validator(path)
    if source is None:
        return utils.Error("unsupported file for validator")

    build = source.build()
    if isinstance(build, BuildError):
        return utils.Error(f"failed to build validator\n{build.msg}")

    return build


def _load_validator(path: Path) -> SourceCode | None:
    if path.suffix == CppSource.SUFFIX:
        return CppSource(path)
    elif path.suffix == PythonSource.SUFFIX:
        return PythonSource(path)
    else:
        return None
