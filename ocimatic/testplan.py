import os
import re
import shlex
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Counter, Dict, List, NoReturn, Optional

import ocimatic
from ocimatic import ui
from ocimatic.runnable import RunError, RunSuccess
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
        self._dataset_dir = dataset_directory

    def validators(self) -> List[Optional[Path]]:
        return [subtask.validator for subtask in self._parse_file()]

    def run(self, stn: Optional[int]) -> None:
        subtasks = self._parse_file()
        cwd = Path.cwd()
        # Run generators with attic/ as the cwd
        os.chdir(self._directory)

        for (i, st) in enumerate(subtasks, 1):
            if stn is None or stn == i:
                st.run()

        if sum(len(st.commands) for st in subtasks) == 0:
            ui.show_message("Warning", 'no commands were executed for the plan.', ui.WARNING)

        os.chdir(cwd)

    def _parse_file(self) -> List['Subtask']:
        subtasks: Dict[int, 'Subtask'] = {}
        st = 0
        tests_in_group: Counter[str] = Counter()
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
                    validator = Path(self._directory,
                                     header_match.group(2)) if header_match.group(2) else None
                    if st + 1 != found_st:
                        ui.fatal_error('line %d: found subtask %d, but subtask %d was expected' %
                                       (lineno, found_st, st + 1))
                    st += 1
                    subtasks[st] = Subtask(self._dataset_dir, st, validator)
                elif cmd_match:
                    if st == 0:
                        ui.fatal_error('line %d: found command before declaring a subtask.' %
                                       lineno)
                    group = cmd_match.group(1)
                    cmd = cmd_match.group(2)
                    args = _parse_args(cmd_match.group(3) or '')

                    tests_in_group[group] += 1
                    command = self._parse_command(group, tests_in_group[group], cmd, args, lineno)
                    subtasks[st].commands.append(command)
                else:
                    ui.fatal_error('line %d: error while parsing line `%s`\n' % (lineno, line))
        return [st for (_, st) in sorted(subtasks.items())]

    def _parse_command(self, group: str, idx: int, cmd: str, args: List[str],
                       lineno: int) -> 'Command':
        if cmd == 'copy':
            if len(args) > 2:
                ui.fatal_error(f'line {lineno}: command `copy` expects exactly one argument.')
            return Copy(group, idx, Path(self._task_directory, args[0]))
        elif cmd == 'echo':
            return Echo(group, idx, args)
        elif Path(cmd).suffix == '.py':
            return Script(group, idx, PythonSource(Path(self._directory, cmd)), args)
        elif Path(cmd).suffix == '.cpp':
            return Script(group, idx, CppSource(Path(self._directory, cmd)), args)
        else:
            _invalid_command(cmd, lineno)


class Subtask:

    def __init__(self, dataset_dir: Path, stn: int, validator: Optional[Path]):
        self._dir = Path(dataset_dir, f'st{stn}')
        self.commands: List['Command'] = []
        self.validator = validator

    def __str__(self) -> str:
        return str(self._dir.name)

    @ui.workgroup()
    def run(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)
        self._dir.mkdir(parents=True, exist_ok=True)
        for cmd in self.commands:
            cmd.run(self._dir)


class Command(ABC):

    def __init__(self, group: str, idx: int):
        self._group = group
        self._idx = idx

    def dst_file(self, dir: Path) -> Path:
        return Path(dir, f'{self._group}-{self._idx}.in')

    @abstractmethod
    def run(self, dst_dir: Path) -> WorkResult:
        raise NotImplementedError("Class %s doesn't implement run()" % (self.__class__.__name__))


class Copy(Command):

    def __init__(self, group: str, idx: int, file: Path):
        super().__init__(group, idx)
        self._file = file

    def __str__(self) -> str:
        return str(self._file.relative_to(ocimatic.config['contest_root']))

    @ui.work('Copy', '{0}')
    def run(self, dst_dir: Path) -> WorkResult:
        if not self._file.exists():
            return WorkResult.fail(short_msg='No such file')
        try:
            shutil.copy(self._file, self.dst_file(dst_dir))
            return WorkResult.success(short_msg='OK')
        except Exception:  # pylint: disable=broad-except
            return WorkResult.fail(short_msg='Error when copying file')


class Echo(Command):

    def __init__(self, group: str, idx: int, args: List[str]):
        super().__init__(group, idx)
        self._args = args

    def __str__(self) -> str:
        return str(self._args)

    @ui.work('Echo', '{0}')
    def run(self, dst_dir: Path) -> WorkResult:
        with self.dst_file(dst_dir).open('w') as test_file:
            test_file.write(' '.join(self._args) + '\n')
            return WorkResult.success(short_msg='Ok')


class Script(Command):
    VALID_EXTENSIONS: List[str] = ['py', 'cpp']

    def __init__(self, group: str, idx: int, script: SourceCode, args: List[str]):
        super().__init__(group, idx)
        self._args = args
        self._script = script

    def __str__(self) -> str:
        args = ' '.join(self._args)
        script = Path(self._script.name).name
        return f'{self._group} ; {script} {args}'

    @ui.work('Gen', '{0}')
    def run(self, dst_dir: Path) -> WorkResult:
        build_result = self._script.build()
        if isinstance(build_result, BuildError):
            return WorkResult.fail(short_msg='Failed to build generator', long_msg=build_result.msg)
        result = build_result.run(out_path=self.dst_file(dst_dir),
                                  args=[self._seed_arg(dst_dir), *self._args])
        match result:
            case RunSuccess(_):
                return WorkResult.success(short_msg='OK')
            case RunError(msg, stderr):
                return WorkResult.fail(short_msg=msg, long_msg=stderr)

    def _seed_arg(self, dir: Path) -> str:
        return f'{dir.name}-{self._group}-{self._idx}'


def _invalid_command(cmd: str, lineno: int) -> NoReturn:
    ui.fatal_error(
        f"line {lineno}: invalid command `{cmd}`\n"
        f"The command should be either `copy`, `echo` or a generator with one of the following extensions `{Script.VALID_EXTENSIONS}`"
    )


def _parse_args(args: str) -> List[str]:
    args = args.strip()
    return [a.encode().decode("unicode_escape") for a in shlex.split(args)]
