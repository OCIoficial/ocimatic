import os
import random
import re
import shlex
import shutil
import string
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile

import ocimatic
from ocimatic import ui
from ocimatic.checkers import Checker
from ocimatic.runnable import Binary, Python3, Runnable
from ocimatic.source_code import (CppSource, JavaSource, PythonSource, SourceCode)
from ocimatic.ui import WorkResult

IN = ".in"
SOL = ".sol"


class Dataset:
    """Test data"""
    def __init__(self, directory: Path, sampledata: List['Test']):
        self._directory = directory
        self._subtasks = [Subtask(d) for d in directory.iterdir() if d.is_dir()]
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


class Test:
    """A single test file. Expected output file may not exist"""
    def __init__(self, in_path: Path, expected_path: Path):
        assert in_path.exists()
        self._in_path = in_path
        self._expected_path = expected_path

    def __str__(self) -> str:
        return str(self._in_path)

    def mtime(self) -> float:
        if self._expected_path.exists():
            return max(self._in_path.stat().st_mtime, self._expected_path.stat().st_mtime)
        return self._in_path.stat().st_mtime

    @ui.work('Gen')
    def gen_expected(self, runnable: Runnable) -> WorkResult:
        """Run binary with this test as input to generate expected output file
        """
        result = runnable.run(self.in_path, self.expected_path, timeout=ocimatic.config['timeout'])
        msg = 'OK' if result.success else result.err_msg
        return WorkResult(success=result.success, short_msg=msg, long_msg=result.stderr)

    @ui.work('Run')
    def run(self, runnable: Runnable, checker: Checker, check: bool = False) -> WorkResult:
        """Run runnable redirect this test as standard input and check output correctness"""
        if not self.expected_path.exists():
            return WorkResult(success=False, short_msg='No expected output file')

        out_path = Path(tempfile.mkstemp()[1])

        result = runnable.run(self.in_path, out_path, timeout=ocimatic.config['timeout'])

        # Execution failed
        if not result.success:
            return WorkResult(success=False, short_msg=result.err_msg, long_msg=result.stderr)

        (st, outcome, checkmsg) = checker.run(self.in_path, self.expected_path, out_path)
        # Checker failed
        if not st:
            msg = 'Failed to run checker: %s' % checkmsg
            return WorkResult(success=st, short_msg=msg)

        st = outcome == 1.0
        if check:
            msg = 'OK' if st else 'FAILED'
            return WorkResult(success=st, short_msg=msg)

        msg = '%s [%.2fs]' % (outcome, result.time)
        if checkmsg:
            msg += ' - %s' % checkmsg
        return WorkResult(success=st, short_msg=msg, long_msg=result.stderr)

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


# FIXME: Refactor class. This should allow to re-enable some pylint checks
class DatasetPlan:
    """Functionality to read and run a plan for generating dataset."""
    def __init__(self,
                 directory: Path,
                 task_directory: Path,
                 dataset_directory: Path,
                 filename: str = 'testplan.txt'):
        self._directory = directory
        self._testplan_path = Path(directory, filename)
        if not self._testplan_path.exists():
            ui.fatal_error('No such file plan for creating dataset: "%s"' % self._testplan_path)
        self._task_directory = task_directory
        self._dataset_directory = dataset_directory

    def test_filepath(self, stn: int, group: int, i: int) -> Path:
        st_dir = Path(self._dataset_directory, 'st%d' % stn)
        return Path(st_dir, '%s-%d.in' % (group, i))

    def validate_input(self, stn: Optional[int]) -> None:
        (_, cmds) = self.parse_file()
        for (i, subtask) in sorted(cmds.items()):
            if stn is None or stn == i:
                self.validate_subtask(i, subtask)

    @ui.workgroup('Subtask {1}')
    def validate_subtask(self, stn: int, subtask: Dict[str, Any]) -> None:
        validator = None
        if subtask['validator']:
            (validator, msg) = self.build_validator(subtask['validator'])
            if validator is None:
                ui.show_message('Warning', 'Failed to build validator: %s' % msg, ui.WARNING)
        else:
            ui.show_message('Info', 'No validator specified', ui.INFO)
        if validator:
            for (group, tests) in sorted(subtask['groups'].items()):
                for (i, _) in enumerate(tests, 1):
                    test_file = self.test_filepath(stn, group, i)
                    self.validate_test_input(test_file, validator)

    @ui.work('Validating', '{1}')
    def validate_test_input(self, test_file: Path, validator: Runnable) -> WorkResult:
        if not test_file.exists():
            return WorkResult(success=False, short_msg='Test file does not exist')
        result = validator.run(test_file, None)
        return WorkResult(success=result.success, short_msg=result.err_msg)

    def build_validator(self, source: str) -> Tuple[Optional[Runnable], str]:
        fp = Path(self._directory, source)
        if not fp.exists():
            return (None, 'File does not exists.')
        if fp.suffix == '.cpp':
            binary = CppSource(fp).build()
            if binary is None:
                return (None, 'Failed to build validator.')
            return (binary, 'OK')
        if fp.suffix == '.py':
            return (Python3(fp), 'OK')
        return (None, 'Not supported source file.')

    def run(self, stn: Optional[int]) -> None:
        (subtasks, cmds) = self.parse_file()
        cwd = Path.cwd()
        # Run generators with attic/ as the cwd
        os.chdir(self._directory)

        for i in range(1, subtasks + 1):
            dir = Path(self._dataset_directory, 'st%d' % i)
            if stn is None or i == stn:
                shutil.rmtree(dir, ignore_errors=True)
                dir.mkdir(parents=True, exist_ok=True)

        if not cmds:
            ui.show_message("Warning", 'no commands were executed for the plan.', ui.WARNING)

        for (i, subtask) in sorted(cmds.items()):
            if stn is None or stn == i:
                self.run_subtask(i, subtask)
        os.chdir(cwd)

    @ui.workgroup('Subtask {1}')
    def run_subtask(self, stn: int, subtask: Dict[str, Any]) -> None:
        groups = subtask['groups']
        for (group, tests) in sorted(groups.items()):
            for (i, test) in enumerate(tests, 1):
                cmd = test['cmd']
                test_file = self.test_filepath(stn, group, i)
                if cmd == 'copy':
                    self.copy(test['file'], test_file)
                elif cmd == 'echo':
                    self.echo(test['args'], test_file)
                else:
                    args = test['args']
                    args.insert(0, '%s-%s-%s' % (stn, group, i))
                    source = Path(self._directory, test['source'])
                    if cmd == 'cpp':
                        self.run_source_code_generator(CppSource(source), args, test_file)
                    elif cmd in 'py':
                        self.run_source_code_generator(PythonSource(source), args, test_file)
                    elif cmd == 'java':
                        self.run_source_code_generator(JavaSource(source.stem, source), args,
                                                       test_file)
                    elif cmd == 'run':
                        bin_path = Path(self._directory, test['bin'])
                        self.run_bin_generator(bin_path, args, test_file)
                    else:
                        ui.fatal_error('unexpected command when running plan: %s ' % cmd)

    @ui.work('Copy', '{1}')
    def copy(self, src: str, dst: Path) -> WorkResult:
        fp = Path(self._task_directory, src)
        if not fp.exists():
            return WorkResult(success=False, short_msg='No such file')
        try:
            shutil.copy(fp, dst)
            (st, msg) = (True, 'OK')
            return WorkResult(success=st, short_msg=msg)
        except Exception:  # pylint: disable=broad-except
            return WorkResult(success=False, short_msg='Error when copying file')

    @ui.work('Echo', '{1}')
    def echo(self, args: List[str], dst: Path) -> WorkResult:
        with dst.open('w') as test_file:
            test_file.write(' '.join(args) + '\n')
            return WorkResult(success=True, short_msg='Ok')

    @ui.work('Gen', '{1}')
    def run_source_code_generator(self, source: SourceCode, args: List[str],
                                  dst: Path) -> WorkResult:
        runnable = source.build()
        if runnable is None:
            return WorkResult(success=False, short_msg='Failed to build generator')
        result = runnable.run(out_path=dst, args=args)
        return WorkResult(success=result.success, short_msg=result.err_msg)

    @ui.work('Gen', '{1}')
    def run_bin_generator(self, bin_path: Path, args: List[str], dst: Path) -> WorkResult:
        if not bin_path.exists():
            return WorkResult(success=False, short_msg='No such file')

        bin = Binary(bin_path)
        if not bin.is_callable():
            return WorkResult(success=False,
                              short_msg='Cannot run file, it may not have correct permissions')
        result = bin.run(None, dst, args)
        return WorkResult(success=result.success, short_msg=result.err_msg, long_msg=result.stderr)

    def parse_file(self) -> Tuple[int, Dict[int, Any]]:
        """
        Args:
            path (FilePath)
        """
        cmds: Dict[int, Any] = {}
        st = 0
        for (lineno, line) in enumerate(self._testplan_path.open('r').readlines(), 1):
            line = line.strip()
            subtask_header = re.compile(r'\s*\[\s*Subtask\s*(\d+)\s*(?:-\s*([^\]\s]+))?\s*\]\s*')
            cmd_line = re.compile(r'\s*([^;\s]+)\s*;\s*(\S+)(:?\s+(.*))?')
            comment = re.compile(r'\s*#.*')

            if not line:
                continue
            if not comment.fullmatch(line):
                header_match = subtask_header.fullmatch(line)
                cmd_match = cmd_line.fullmatch(line)
                if header_match:
                    found_st = int(header_match.group(1))
                    validator = header_match.group(2)
                    if st + 1 != found_st:
                        ui.fatal_error('line %d: found subtask %d, but subtask %d was expected' %
                                       (lineno, found_st, st + 1))
                    st += 1
                    cmds[st] = {'validator': validator, 'groups': {}}
                elif cmd_match:
                    if st == 0:
                        ui.fatal_error('line %d: found command before declaring a subtask.' %
                                       lineno)
                    group = cmd_match.group(1)
                    cmd = cmd_match.group(2)
                    args = _parse_args(cmd_match.group(3) or '')
                    if group not in cmds[st]['groups']:
                        cmds[st]['groups'][group] = []

                    if cmd == 'copy':
                        if len(args) > 2:
                            ui.fatal_error('line %d: command copy expects exactly one argument.' %
                                           lineno)
                        cmds[st]['groups'][group].append({
                            'cmd': 'copy',
                            'file': args[0],
                        })
                    elif cmd == 'echo':
                        cmds[st]['groups'][group].append({'cmd': 'echo', 'args': args})
                    else:
                        f = Path(self._directory, cmd)
                        if f.suffix in ['.cpp', '.java', '.py']:
                            cmds[st]['groups'][group].append({
                                'cmd': f.suffix[1:],
                                'source': cmd,
                                'args': args
                            })
                        else:
                            cmds[st]['groups'][group].append({
                                'cmd': 'run',
                                'bin': cmd,
                                'args': args
                            })
                else:
                    ui.fatal_error('line %d: error while parsing line `%s`\n' % (lineno, line))
        return (st, cmds)


def _parse_args(args: str) -> List[str]:
    args = args.strip()
    return [a.encode().decode("unicode_escape") for a in shlex.split(args)]
