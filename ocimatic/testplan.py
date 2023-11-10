from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from ocimatic import utils
from ocimatic.runnable import ret_code_to_str
from ocimatic.source_code import BuildError, CppSource, PythonSource, SourceCode
from ocimatic.utils import Error

# https://en.wikipedia.org/wiki/C0_and_C1_control_codes#Field_separators
FS = chr(28)


class Testplan:
    """Functionality to read and run a plan for generating dataset."""

    def __init__(
        self,
        directory: Path,
        task_directory: Path,
        dataset_directory: Path,
        filename: str = "testplan.txt",
    ) -> None:
        self._directory = directory
        self._testplan_path = Path(directory, filename)
        if not self._testplan_path.exists():
            utils.fatal_error(
                f'File not found: "{self._testplan_path}"',
            )
        self._task_directory = task_directory
        self._dataset_dir = dataset_directory

        subtasks = self._parse_file()
        if isinstance(subtasks, Error):
            utils.writeln(
                f"Error when parsing testplan in `{utils.relative_to_cwd(self._testplan_path)}`",
                utils.ERROR,
            )
            utils.writeln(subtasks.msg, utils.ERROR)
            sys.exit(1)

        self._subtasks = subtasks

    @property
    def subtasks(self) -> int:
        return len(self._subtasks)

    def validators(self) -> list[Path | None]:
        return [subtask.validator for subtask in self._subtasks]

    def includes(self, stn: int) -> list[Include]:
        return self._subtasks[stn - 1].includes

    def extract_group(self, test_file: Path) -> str | None:
        return _Command.extract_group(test_file)

    def run(self, stn: int | None) -> utils.Status:
        cwd = Path.cwd()
        # Run generators with `testplan/` as the cwd
        os.chdir(self._directory)

        status = utils.Status.success
        for i, st in enumerate(self._subtasks, 1):
            if stn is not None and stn != i:
                continue
            status &= st.run()

        if sum(len(st.commands) for st in self._subtasks) == 0:
            utils.show_message(
                "Warning",
                "no commands were executed for the plan.",
                utils.WARNING,
            )

        os.chdir(cwd)

        return status

    def _parse_file(self) -> list[_SubtaskPlan] | Error:
        comment_re = re.compile(r"\s*#.*")
        subtasks_map: dict[int, _SubtaskPlan] = {}
        stn = 0
        for lineno, line in enumerate(self._testplan_path.open("r").readlines(), 1):
            line = line.strip()

            if not line:
                continue
            if comment_re.fullmatch(line):
                continue

            if m := _SubtaskPlan.RE.fullmatch(line):
                found_st = int(m.group(1))
                validator = Path(self._directory, m.group(2)) if m.group(2) else None
                stn += 1
                if stn != found_st:
                    return Error(
                        f"line {lineno}: found [Subtask {found_st}], but [Subtask {stn}] was expected",
                    )
                subtasks_map[stn] = _SubtaskPlan(self._dataset_dir, stn, validator)
            elif m := _Command.RE.fullmatch(line):
                if stn == 0:
                    return Error(
                        f"line {lineno}: found command before declaring a subtask.",
                    )
                group_str = m.group(1)
                group = _GroupName.parse(group_str)
                if not group:
                    return Error(
                        f"line {lineno}: invalid group name `{group_str}`. The group name should "
                        f"match the regex `{_GroupName.RE.pattern}`.",
                    )
                cmd = m.group(2)
                args = _parse_args(m.group(3) or "")

                command = self._parse_command(
                    group,
                    cmd,
                    args,
                    lineno,
                )
                if isinstance(command, Error):
                    return command
                subtasks_map[stn].commands.append(command)
            elif m := Include.RE.fullmatch(line):
                subtasks_map[stn].includes.append(
                    Include(
                        stn=int(m.group(2)),
                        pattern=m.group(1),
                        lineno=lineno,
                    ),
                )
            else:
                return Error(
                    f"line {lineno}: invalid line `{line}`\n",
                )
        subtasks = [st for (_, st) in sorted(subtasks_map.items())]
        validated = Testplan._validate(subtasks)
        if isinstance(validated, Error):
            return validated
        return subtasks

    def _parse_command(
        self,
        group: _GroupName,
        cmd: str,
        args: list[str],
        lineno: int,
    ) -> _Command | Error:
        if cmd == "copy":
            if len(args) > 2:
                return Error(
                    f"line {lineno}: the `copy` command expects exactly one argument.",
                )
            return _Copy(group, self._task_directory, args[0])
        elif cmd == "echo":
            return _Echo(group, args)
        elif Path(cmd).suffix == ".py":
            return _Script(group, Path(self._directory, cmd), "py", args)
        elif Path(cmd).suffix == ".cpp":
            return _Script(group, Path(self._directory, cmd), "cpp", args)
        else:
            return _invalid_command_err(cmd, lineno)

    @staticmethod
    def _validate(subtasks: list[_SubtaskPlan]) -> Error | None:
        for i, st in enumerate(subtasks):
            for include in st.includes:
                lineno = include.lineno
                if include.stn not in range(1, len(subtasks) + 1):
                    return Error(
                        f"line {lineno}: invalid subtask {include.stn}: `{include}`",
                    )
                if include.stn == i + 1:
                    return Error(
                        f"line {lineno}: cannot include tests from the same subtask: `{include}`",
                    )
        return None


@dataclass(kw_only=True, frozen=True, slots=True)
class Include:
    """An include directive can be used to include tests from another subtask.

    This is not fully impelemented yet.
    """

    RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\s*@\s*include\s*([^\s]+)\s+from\s+subtask\s*(\d+)\s*",
    )

    stn: int
    pattern: str
    lineno: int

    def __str__(self) -> str:
        return f"@include {self.pattern} from subtask {self.stn}"


@dataclass(kw_only=True, frozen=True, slots=True)
class _GroupName:
    RE: ClassVar[re.Pattern[str]] = re.compile(r"[a-zA-Z0-9_-]+")

    name: str

    @staticmethod
    def parse(name: str) -> _GroupName | None:
        if not _GroupName.RE.fullmatch(name):
            return None
        return _GroupName(name=name)

    def __str__(self) -> str:
        return self.name


class _SubtaskPlan:
    RE = re.compile(r"\s*\[\s*Subtask\s*(\d+)\s*(?:-\s*([^\]\s]+))?\s*\]\s*")

    def __init__(self, dataset_dir: Path, stn: int, validator: Path | None) -> None:
        self._dir = Path(dataset_dir, f"st{stn}")
        self.commands: list[_Command] = []
        self.includes: list[Include] = []
        self.validator = validator

    def __str__(self) -> str:
        return str(self._dir.name)

    @utils.hd2("{0}")
    def run(self) -> utils.Status:
        shutil.rmtree(self._dir, ignore_errors=True)
        self._dir.mkdir(parents=True, exist_ok=True)

        status = utils.Status.success
        tests_in_group: Counter[_GroupName] = Counter()
        for cmd in self.commands:
            status &= cmd.run(self._dir, tests_in_group).status
        return status


class _Command(ABC):
    RE = re.compile(r"\s*([^;\s]+)\s*;\s*(\S+)(:?\s+(.*))?")
    FILE_RE = re.compile(r"(.*)-(\d+).in")

    def __init__(self, group: _GroupName) -> None:
        self._group = group

    def dst_file(self, directory: Path, idx: int) -> Path:
        return Path(directory, self.add_group(idx))

    def add_group(self, idx: int) -> str:
        name = f"{self._group}-{idx}.in"
        assert _Command.FILE_RE.fullmatch(name) is not None
        return name

    @staticmethod
    def extract_group(test_file: Path) -> str | None:
        m = _Command.FILE_RE.fullmatch(test_file.name)
        return m.group(1) if m else None

    @abstractmethod
    def run(self, dst_dir: Path, tests_in_group: Counter[_GroupName]) -> utils.Result:
        raise NotImplementedError(
            f"Class {self.__class__.__name__} doesn't implement run()",
        )


class _Copy(_Command):
    magic_check = re.compile("([*?[])")

    def __init__(self, group: _GroupName, dir: Path, pattern: str) -> None:
        super().__init__(group)
        self._dir = dir
        self._pattern = pattern

    def __str__(self) -> str:
        return self._pattern

    @utils.work("Copy", "{0}")
    def run(self, dst_dir: Path, tests_in_group: Counter[_GroupName]) -> utils.Result:
        files = list(self._dir.glob(self._pattern))
        if not files:
            msg = "No file matches the pattern" if self.has_magic() else "No such file"
            return utils.Result.fail(short_msg=msg)
        try:
            for file in files:
                idx = _next_idx_in_group(self._group, tests_in_group)
                shutil.copy(file, self.dst_file(dst_dir, idx))
            return _success_with_count_result(len(files))
        except Exception:  # pylint: disable=broad-except
            return utils.Result.fail(short_msg="Error when copying file")

    def has_magic(self) -> bool:
        return _Copy.magic_check.search(self._pattern) is not None


class _Echo(_Command):
    def __init__(self, group: _GroupName, args: list[str]) -> None:
        super().__init__(group)
        self._args = args

    def __str__(self) -> str:
        return str(self._args)

    @utils.work("Echo", "{0}")
    def run(self, dst_dir: Path, tests_in_group: Counter[_GroupName]) -> utils.Result:
        idx = _next_idx_in_group(self._group, tests_in_group)
        with self.dst_file(dst_dir, idx).open("w") as test_file:
            test_file.write(" ".join(self._args) + "\n")
            return _success_with_count_result(1)


class _Script(_Command):
    VALID_EXTENSIONS = Literal["py", "cpp"]

    def __init__(
        self,
        group: _GroupName,
        path: Path,
        ext: VALID_EXTENSIONS,
        args: list[str],
    ) -> None:
        super().__init__(group)
        self._args = args
        self._script_path = path
        self._ext = ext

    def __str__(self) -> str:
        args = " ".join(self._args)
        script = self._script_path.name
        return f"{self._group} ; {script} {args}"

    @utils.work("Gen", "{0}")
    def run(self, dst_dir: Path, tests_in_group: Counter[_GroupName]) -> utils.Result:
        script = self._load_script()
        if not script:
            return utils.Result.fail(short_msg="Script file not found")
        build_result = script.build()
        if isinstance(build_result, BuildError):
            return utils.Result.fail(
                short_msg="Failed to build generator",
                long_msg=build_result.msg,
            )

        count = 0
        idx = _next_idx_in_group(self._group, tests_in_group)
        # We seed the script with the first `idx`, this guarantees it will be different next time.
        process = build_result.spawn([self._seed_arg(dst_dir, idx), *self._args])
        current_file = None
        try:
            assert process.stdout
            while char := process.stdout.read(1):
                if char == FS:
                    if current_file:
                        current_file.close()
                        current_file = None
                else:
                    if current_file is None:
                        count += 1
                        current_file = self.dst_file(dst_dir, idx).open("w")
                        idx = _next_idx_in_group(self._group, tests_in_group)
                    current_file.write(char)
        finally:
            if current_file:
                current_file.close()

        if count == 0:
            return utils.Result.fail(short_msg="generator didn't produce any output")

        ret = process.wait()
        if ret == 0:
            return _success_with_count_result(count)
        else:
            msg = ret_code_to_str(ret)
            long_msg = process.stderr.read() if process.stderr else None
            return utils.Result.fail(short_msg=msg, long_msg=long_msg)

    def _load_script(self) -> SourceCode | None:
        if not self._script_path.exists():
            return None
        if self._ext == "py":
            return PythonSource(self._script_path)
        elif self._ext == "cpp":
            return CppSource(self._script_path)

    def _seed_arg(self, directory: Path, idx: int) -> str:
        return f"{directory.name}-{self._group}-{idx}"


def _invalid_command_err(cmd: str, lineno: int) -> Error:
    from typing import get_args

    extensions = get_args(_Script.VALID_EXTENSIONS)
    return Error(
        f"line {lineno}: invalid command `{cmd}`\n"
        f"The command should be either `copy`, `echo` or a generator with one of the following extensions {extensions}",
    )


def _success_with_count_result(count: int) -> utils.Result:
    assert count > 0
    if count == 1:
        return utils.Result.success(short_msg="1 test case generated")
    else:
        return utils.Result.success(short_msg=f"{count} test cases generated")


def _parse_args(args: str) -> list[str]:
    args = args.strip()
    return [a.encode().decode("unicode_escape") for a in shlex.split(args)]


def _next_idx_in_group(group: _GroupName, tests_in_group: Counter[_GroupName]) -> int:
    tests_in_group[group] += 1
    return tests_in_group[group]
