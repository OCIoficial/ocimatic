import random
import shutil
import string
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
from zipfile import ZipFile

import ocimatic
from ocimatic import ui
from ocimatic.checkers import Checker, CheckerError
from ocimatic.runnable import RunError, Runnable, RunSuccess
from ocimatic.source_code import (BuildError, CppSource, PythonSource,
                                  SourceCode)
from ocimatic.ui import WorkResult

IN = ".in"
SOL = ".sol"


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
            check: bool = False,
            sample: bool = False) -> None:
        for subtask in self._subtasks:
            subtask.run(runnable, checker, check=check)
        if sample:
            self._sampledata.run(runnable, checker, check=check)

    def validate(self, validators: List[Optional[Path]], stn: Optional[int]) -> None:
        assert len(validators) == len(self._subtasks)
        for (i, (subtask, validator)) in enumerate(zip(self._subtasks, validators), 1):
            if stn is None or stn == i:
                subtask.validate(validator)

    def __str__(self) -> str:
        return f"Dataset({self._directory})"

    def mtime(self) -> float:
        mtime = -1.0
        for subtask in self._subtasks:
            mtime = max(mtime, subtask.mtime())
        return mtime

    def compress(self, random_sort: bool = False) -> bool:
        """Compress all test cases in the dataset into a single zip file.
        The basename of the corresponding subtask subdirectory is prepended
        to each file.
        """
        path = Path(self._directory, 'data.zip')
        if path.exists() and path.stat().st_mtime >= self.mtime():
            return True

        with ZipFile(path, 'w') as zip:
            compressed = 0
            for subtask in self._subtasks:
                compressed += subtask.write_to_zip(zip, random_sort)

            if compressed == 0:
                ui.show_message("Warning", "no files in dataset", ui.WARNING)
        return True

    def count(self) -> List[int]:
        return [st.count() for st in self._subtasks]

    def normalize(self) -> None:
        for subtask in self._subtasks:
            subtask.normalize()
        self._sampledata.normalize()


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
    def run(self, runnable: Runnable, checker: Checker, check: bool = False) -> None:
        for test in self._tests:
            test.run(runnable, checker, check=check)

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
    def run(self, runnable: Runnable, checker: Checker, check: bool = False) -> WorkResult:
        """Run runnable redirect this test as standard input and check output correctness"""
        if not self.expected_path.exists():
            return WorkResult(success=False, short_msg='No expected output file')

        out_path = Path(tempfile.mkstemp()[1])

        result = runnable.run(self.in_path, out_path, timeout=ocimatic.config['timeout'])

        # Execution failed
        if isinstance(result, RunError):
            return WorkResult(success=False, short_msg=result.msg, long_msg=result.stderr)

        checker_result = checker.run(self.in_path, self.expected_path, out_path)

        # Checker failed
        if isinstance(checker_result, CheckerError):
            msg = f'Failed to run checker: `{checker_result.msg}`'
            return WorkResult(success=False, short_msg=msg)

        outcome = checker_result.outcome
        st = outcome == 1.0
        if check:
            msg = 'OK' if st else 'FAILED'
            return WorkResult(success=st, short_msg=msg)

        msg = f'{outcome} [{result.time:.2f}s]'
        if checker_result.msg is not None:
            msg += ' - %s' % checker_result.msg
        return WorkResult(success=st, short_msg=msg)

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
