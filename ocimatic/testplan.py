import os
import re
import shlex
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple

from ocimatic import ui
from ocimatic.runnable import Python3, Runnable, RunSuccess
from ocimatic.source_code import (BuildError, CppSource, PythonSource, SourceCode)
from ocimatic.ui import WorkResult


class Testplan:
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
        if isinstance(result, RunSuccess):
            return WorkResult(success=True, short_msg='OK')
        else:
            return WorkResult(success=False, short_msg=result.msg, long_msg=result.stderr)

    def build_validator(self, source: str) -> Tuple[Optional[Runnable], str]:
        fp = Path(self._directory, source)
        if not fp.exists():
            return (None, 'File does not exists.')
        if fp.suffix == '.cpp':
            build_result = CppSource(fp).build()
            if isinstance(build_result, BuildError):
                return (None, 'Failed to build validator.')
            return (build_result, 'OK')
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
            for (i, cmd) in enumerate(tests, 1):
                seed_arg = '{stn}-{group}-{i}'
                test_file = self.test_filepath(stn, group, i)
                cmd.run(seed_arg, test_file)

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

                    command = self.parse_command(cmd, args, lineno)
                    cmds[st]['groups'][group].append(command)
                else:
                    ui.fatal_error('line %d: error while parsing line `%s`\n' % (lineno, line))
        return (st, cmds)

    def parse_command(self, cmd: str, args: List[str], lineno: int) -> 'Command':
        if cmd == 'copy':
            if len(args) > 2:
                ui.fatal_error(f'line {lineno}: command `copy` expects exactly one argument.')
            return Copy(Path(self._task_directory, args[0]))
        elif cmd == 'echo':
            return Echo(args)
        elif Path(cmd).suffix == '.py':
            return Script(PythonSource(Path(self._directory, cmd)), args)
        elif Path(cmd).suffix == '.cpp':
            return Script(CppSource(Path(self._directory, cmd)), args)
        else:
            _invalid_command(cmd, lineno)


class Command(ABC):
    @abstractmethod
    def run(self, seed_arg: str, dst: Path) -> WorkResult:
        raise NotImplementedError("Class %s doesn't implement run()" % (self.__class__.__name__))


class Copy(Command):
    def __init__(self, file: Path):
        self._file = file

    def __str__(self) -> str:
        return str(self._file)

    @ui.work('Copy', '{0}')
    def run(self, seed_arg: str, dst: Path) -> WorkResult:
        del seed_arg
        if not self._file.exists():
            return WorkResult(success=False, short_msg='No such file')
        try:
            shutil.copy(self._file, dst)
            (st, msg) = (True, 'OK')
            return WorkResult(success=st, short_msg=msg)
        except Exception:  # pylint: disable=broad-except
            return WorkResult(success=False, short_msg='Error when copying file')


class Echo(Command):
    def __init__(self, args: List[str]):
        self._args = args

    def __str__(self) -> str:
        return str(self._args)

    @ui.work('Echo', '{0}')
    def run(self, seed_arg: str, dst: Path) -> WorkResult:
        del seed_arg
        with dst.open('w') as test_file:
            test_file.write(' '.join(self._args) + '\n')
            return WorkResult(success=True, short_msg='Ok')


class Script(Command):
    VALID_EXTENSIONS: List[str] = ['py', 'cpp']

    def __init__(self, script: SourceCode, args: List[str]):
        self._args = args
        self._script = script

    def __str__(self) -> str:
        return str(self._script)

    @ui.work('Gen', '{0}')
    def run(self, seed_arg: str, dst: Path) -> WorkResult:
        build_result = self._script.build()
        if isinstance(build_result, BuildError):
            return WorkResult(success=False,
                              short_msg='Failed to build generator',
                              long_msg=build_result.msg)
        result = build_result.run(out_path=dst, args=[seed_arg, *self._args])
        if isinstance(result, RunSuccess):
            return WorkResult(success=True, short_msg='OK')
        else:
            return WorkResult(success=False, short_msg=result.msg, long_msg=result.stderr)


def _invalid_command(cmd: str, lineno: int) -> NoReturn:
    ui.fatal_error(
        f"line {lineno}: invalid command `{cmd}`\n"
        f"The command should be either `copy`, `echo` or a generator with one of the following extensions `{Script.VALID_EXTENSIONS}`"
    )


def _parse_args(args: str) -> List[str]:
    args = args.strip()
    return [a.encode().decode("unicode_escape") for a in shlex.split(args)]
