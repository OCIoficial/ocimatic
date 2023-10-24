import itertools
import random
import shutil
import statistics
import string
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional
from zipfile import ZipFile

import ocimatic
from ocimatic import ui
from ocimatic.checkers import Checker, CheckerError, CheckerSuccess
from ocimatic.runnable import RunError, Runnable, RunSuccess
from ocimatic.source_code import (BuildError, CppSource, PythonSource, SourceCode)
from ocimatic.ui import WorkResult

IN = ".in"
SOL = ".sol"


class TestResult:

    @dataclass
    class Success:
        checker_result: CheckerSuccess
        run_result: RunSuccess
        check_mode: bool

        def running_time(self) -> float:
            return self.run_result.time

        def into_work_result(self) -> WorkResult:
            outcome = self.checker_result.outcome
            st = outcome == 1.0
            if self.check_mode:
                msg = 'OK' if st else 'FAILED'
                return WorkResult(success=st, short_msg=msg)

            msg = f'{outcome} [{self.run_result.time:.3f}s]'
            if self.checker_result.msg is not None:
                msg += ' - %s' % self.checker_result.msg
            return WorkResult(success=st, short_msg=msg)

    @dataclass
    class CheckerError:
        checker_result: CheckerError
        run_result: RunSuccess

        def running_time(self) -> float:
            return self.run_result.time

        def into_work_result(self) -> WorkResult:
            msg = f'Failed to run checker: `{self.checker_result.msg}`'
            return WorkResult(success=False, short_msg=msg)

    @dataclass
    class RuntimeError:
        run_result: RunError

        def into_work_result(self) -> WorkResult:
            return WorkResult(success=False,
                              short_msg=self.run_result.msg,
                              long_msg=self.run_result.stderr)

    @dataclass
    class NoExpectedOutput:

        def into_work_result(self) -> WorkResult:
            return WorkResult(success=False, short_msg='No expected output file')

    T = Success | RuntimeError | CheckerError | NoExpectedOutput


class Test:
    """A single test file. Expected output file may not exist"""

    def __init__(self, in_path: Path, expected_path: Path):
        assert in_path.exists()
        self._in_path = in_path
        self._expected_path = expected_path

    def __str__(self) -> str:
        return str(self._in_path.relative_to(ocimatic.config['contest_root']))

    def mtime(self) -> float:
        if self._expected_path.exists():
            return max(self._in_path.stat().st_mtime, self._expected_path.stat().st_mtime)
        return self._in_path.stat().st_mtime

    @ui.work('Validate', '{0}')
    def validate(self, validator: Runnable) -> WorkResult:
        result = validator.run(self._in_path, None)
        if isinstance(result, RunSuccess):
            return WorkResult(success=True, short_msg='OK')
        else:
            return WorkResult(success=False, short_msg=result.msg, long_msg=result.stderr)

    @ui.work('Gen')
    def gen_expected(self, runnable: Runnable) -> WorkResult:
        """Run binary with this test as input to generate expected output file
        """
        result = runnable.run(self.in_path, self.expected_path, timeout=ocimatic.config['timeout'])
        if isinstance(result, RunSuccess):
            return WorkResult(success=True, short_msg='OK')
        else:
            return WorkResult(success=False, short_msg=result.msg, long_msg=result.stderr)

    @ui.work('Run')
    def run(self, runnable: Runnable, checker: Checker, check_mode: bool = False) -> TestResult.T:
        """Run runnable redirect this test as standard input and check output correctness"""
        if not self.expected_path.exists():
            return TestResult.NoExpectedOutput()

        out_path = Path(tempfile.mkstemp()[1])

        run_result = runnable.run(self.in_path, out_path, timeout=ocimatic.config['timeout'])

        # Execution failed
        if isinstance(run_result, RunError):
            return TestResult.RuntimeError(run_result)

        checker_result = checker.run(self.in_path, self.expected_path, out_path)

        # Checker failed
        if isinstance(checker_result, CheckerError):
            return TestResult.CheckerError(checker_result, run_result)

        return TestResult.Success(checker_result, run_result, check_mode)

    @property
    def in_path(self) -> Path:
        return self._in_path

    @property
    def expected_path(self) -> Path:
        return self._expected_path

    @ui.work('Normalize')
    def normalize(self) -> WorkResult:
        if not shutil.which('dos2unix'):
            return WorkResult(success=False, short_msg='Cannot find dos2unix')
        if not shutil.which('sed'):
            return WorkResult(success=False, short_msg='Cannot find sed')
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
        return WorkResult(success=st == 0, short_msg='OK' if st == 0 else 'FAILED')


class TestGroup:

    def __init__(self, name: str, tests: List['Test']):
        self._name = name
        self._tests = tests

    def write_to_zip(self, zip: ZipFile, random_sort: bool = False) -> int:
        copied = 0
        for test in self._tests:
            if test.expected_path.exists():
                # Sort testcases withing a subtask randomly
                if random_sort:
                    choices = string.ascii_lowercase
                    rnd_str = ''.join(random.choice(choices) for _ in range(3))
                    in_name = "%s-%s-%s" % (self._name, rnd_str, test.in_path.name)
                    sol_name = "%s-%s-%s" % (self._name, rnd_str, test.expected_path.name)
                else:
                    in_name = "%s-%s" % (self._name, test.in_path.name)
                    sol_name = "%s-%s" % (self._name, test.expected_path.name)
                zip.write(test.in_path, in_name)
                zip.write(test.expected_path, sol_name)
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

    @ui.workgroup()
    def run(self,
            runnable: Runnable,
            checker: Checker,
            check_mode: bool = False) -> List[TestResult.T]:
        results = []
        for test in self._tests:
            results.append(test.run(runnable, checker, check_mode=check_mode))
        return results

    @ui.workgroup()
    def gen_expected(self, runnable: Runnable) -> None:
        for test in self._tests:
            test.gen_expected(runnable)

    def __str__(self) -> str:
        return self._name


class Subtask(TestGroup):

    def __init__(self, directory: Path):
        super().__init__(directory.name,
                         [Test(f, f.with_suffix(SOL)) for f in directory.glob(f'*{IN}')])

    @ui.workgroup('{0}')
    def validate(self, validator: Optional[Path]) -> None:
        if validator is None:
            ui.show_message('Info', 'No validator specified', ui.INFO)
            return
        source: SourceCode
        if validator.suffix == '.cpp':
            source = CppSource(validator)
        elif validator.suffix == '.py':
            source = PythonSource(validator)
        else:
            ui.show_message('Warning', 'Unsupported file for validator', ui.WARNING)
            return
        build = source.build()
        if isinstance(build, BuildError):
            ui.show_message('Warning', f'Failed to build validator\n{build.msg}', ui.WARNING)
            return
        for test in self._tests:
            test.validate(build)


@dataclass
class DatasetResult:
    subtasks: Dict[str, List[TestResult.T]]
    sample: Optional[List[TestResult.T]]

    def running_times(self, include_sample: bool = False) -> Iterator[float]:
        """Returns running times of all successful runs"""
        tests: Iterable[TestResult.T] = (t for st in self.subtasks.values() for t in st)
        if include_sample and self.sample is not None:
            tests = itertools.chain(tests, self.sample)
        for test in tests:
            if isinstance(test, TestResult.Success) or isinstance(test, TestResult.CheckerError):
                yield test.running_time()


class Dataset:
    """Test data"""

    def __init__(self, directory: Path, sampledata: List['Test']):
        self._directory = directory
        if directory.exists():
            self._subtasks = [Subtask(d) for d in sorted(directory.iterdir()) if d.is_dir()]
        else:
            self._subtasks = []
        self._sampledata = TestGroup('sample', sampledata)

    def gen_expected(self, runnable: Runnable, sample: bool = False) -> None:
        for subtask in self._subtasks:
            subtask.gen_expected(runnable)
        if sample:
            self._sampledata.gen_expected(runnable)

    def run(self,
            runnable: Runnable,
            checker: Checker,
            check_mode: bool = False,
            run_on_sample_data: bool = False) -> DatasetResult:
        subtasks: Dict[str, List[TestResult.T]] = {}
        for subtask in self._subtasks:
            result = subtask.run(runnable, checker, check_mode=check_mode)
            subtasks[str(subtask)] = result

        sample = None
        if run_on_sample_data:
            sample = self._sampledata.run(runnable, checker, check_mode=check_mode)
        return DatasetResult(subtasks, sample)

    def validate(self, validators: List[Optional[Path]], stn: Optional[int]) -> None:
        assert len(validators) == len(self._subtasks)
        for (i, (subtask, validator)) in enumerate(zip(self._subtasks, validators), 1):
            if stn is None or stn == i:
                subtask.validate(validator)

    def __str__(self) -> str:
        return f"{self._directory}"

    def mtime(self) -> float:
        mtime = -1.0
        for subtask in self._subtasks:
            mtime = max(mtime, subtask.mtime())
        return mtime

    @ui.work('ZIP')
    def compress(self, random_sort: bool = False) -> ui.WorkResult:
        """Compress all test cases in the dataset into a single zip file.
        The basename of the corresponding subtask subdirectory is prepended
        to each file.
        """
        path = Path(self._directory, 'data.zip')

        with ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zip:
            compressed = 0
            for subtask in self._subtasks:
                compressed += subtask.write_to_zip(zip, random_sort)

            if compressed == 0:
                path.unlink()
                return ui.WorkResult(success=False, short_msg='EMPTY DATASET')
        return ui.WorkResult(success=True, short_msg="OK")

    def count(self) -> List[int]:
        return [st.count() for st in self._subtasks]

    def normalize(self) -> None:
        for subtask in self._subtasks:
            subtask.normalize()
        self._sampledata.normalize()
