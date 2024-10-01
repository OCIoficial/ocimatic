from __future__ import annotations

import re
import shlex
import shutil
import sys
import typing
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from ocimatic import utils
from ocimatic.runnable import ret_code_to_str
from ocimatic.source_code import BuildError, CppSource, PythonSource, SourceCode
from ocimatic.utils import Error, SortedDict, Stn

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
        self._testplan_path = directory / filename
        if not self._testplan_path.exists():
            utils.fatal_error(
                f'File not found: "{self._testplan_path}"',
            )
        self._task_directory = task_directory
        self._dataset_dir = dataset_directory

        subtasks = self._parse_file()
        if isinstance(subtasks, ParseError):
            utils.writeln(
                f"Error when parsing testplan in `{utils.relative_to_cwd(self._testplan_path)}`",
                utils.ERROR,
            )
            utils.writeln(f"{subtasks}", utils.ERROR)
            sys.exit(1)

        self._subtasks = subtasks

    @property
    def subtasks(self) -> int:
        return len(self._subtasks)

    def validators(self) -> list[Path | None]:
        return [st.validator for st in self._subtasks.values()]

    def ancestors_of(self, stn: Stn) -> list[Stn]:
        visited: set[Stn] = set()

        def dfs(sti: Stn) -> None:
            visited.add(sti)

            for extends in self._subtasks[sti].extends:
                if extends.stn not in visited:
                    dfs(extends.stn)

        dfs(stn)
        visited.remove(stn)
        return sorted(visited)

    def parents_of(self, stn: Stn) -> list[Stn]:
        return sorted(extends.stn for extends in self._subtasks[stn].extends)

    def run(self, stn: Stn | None) -> utils.Status:
        status = utils.Status.success
        for sti, st in self._subtasks.items():
            if stn is not None and stn != sti:
                continue
            status &= st.run()

        if sum(len(st.commands) for st in self._subtasks.values()) == 0:
            utils.show_message(
                "Warning",
                "no commands were executed for the plan.",
                utils.WARNING,
            )

        return status

    def _parse_file(self) -> SortedDict[Stn, _Subtask] | ParseError:
        comment_re = re.compile(r"\s*#.*")
        subtasks: SortedDict[Stn, _Subtask] = SortedDict()
        sti = 0
        for lineno, line in enumerate(self._testplan_path.open("r").readlines(), 1):
            line = line.strip()

            if not line:
                continue
            if comment_re.fullmatch(line):
                continue

            if m := _Subtask.RE.fullmatch(line):
                found_st = int(m.group(1))
                validator = Path(self._directory, m.group(2)) if m.group(2) else None
                sti += 1
                if sti != found_st:
                    return ParseError(
                        lineno=lineno,
                        msg=f"found [Subtask {found_st}], but [Subtask {sti}] was expected",
                    )
                subtasks[Stn(sti)] = _Subtask(self._dataset_dir, sti, validator)
            elif m := _Command.RE.fullmatch(line):
                if sti == 0:
                    return ParseError(
                        lineno=lineno,
                        msg="found command before declaring a subtask.",
                    )
                group_str = m.group(1)
                group = _GroupName.parse(group_str)
                if not group:
                    return ParseError(
                        lineno=lineno,
                        msg="invalid group name `{group_str}`. The group name should "
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
                if isinstance(command, ParseError):
                    return command
                subtasks[Stn(sti)].commands.append(command)
            elif m := _Extends.RE.fullmatch(line):
                subtasks[Stn(sti)].extends.append(
                    _Extends(
                        stn=Stn(int(m.group(1))),
                        lineno=lineno,
                    ),
                )
            else:
                return ParseError(lineno=lineno, msg=f"invalid line `{line}`")
        validated = Testplan._validate_extends_graph(subtasks)
        if isinstance(validated, ParseError):
            return validated
        return subtasks

    def _parse_command(
        self,
        group: _GroupName,
        cmd: str,
        args: list[str],
        lineno: int,
    ) -> _Command | ParseError:
        if cmd == "copy":
            if len(args) > 2:
                return ParseError(
                    lineno=lineno,
                    msg="the `copy` command expects exactly one argument.",
                )
            return _Copy(group, self._task_directory, args[0])
        elif cmd == "echo":
            return _Echo(group, args)
        elif (ext := Path(cmd).suffix) in (".py", ".cpp"):
            return _Script(
                group,
                Path(self._directory, cmd),
                ext,  # type: ignore
                args,
                self._directory,
            )
        else:
            return ParseError(lineno=lineno, msg=_invalid_command_err_msg(cmd))

    @staticmethod
    def _validate_extends_graph(
        subtasks: SortedDict[Stn, _Subtask],
    ) -> ParseError | None:
        for sti, st in subtasks.items():
            seen: set[Stn] = set()
            for extends in st.extends:
                lineno = extends.lineno
                if extends.stn in seen:
                    return ParseError(
                        lineno=lineno,
                        msg=f"cannot extends twice from the same subtask: `{extends}`",
                    )
                if extends.stn not in subtasks:
                    return ParseError(
                        lineno=lineno,
                        msg=f"invalid subtask {extends.stn}: `{extends}`",
                    )
                if extends.stn == sti:
                    return ParseError(
                        lineno=lineno,
                        msg=f"a subtask cannot extend itself: `{subtasks}`",
                    )
                seen.add(extends.stn)

        if _has_cycles(subtasks):
            return ParseError(msg="the extends graph contains cycles")

        return None


@dataclass(kw_only=True, frozen=True, slots=True)
class ParseError:
    lineno: int | None = None
    msg: str

    def __str__(self) -> str:
        if self.lineno:
            return f"line {self.lineno}: {self.msg}"
        else:
            return self.msg


@dataclass(kw_only=True, frozen=True, slots=True)
class _Extends:
    """An extends directive can be used to include all tests from another subtask."""

    RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\s*@\s*extends\s*subtask\s*(\d+)\s*",
    )

    stn: Stn
    lineno: int

    def __str__(self) -> str:
        return f"@extends subtask {self.stn}"


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


class _Subtask:
    RE = re.compile(r"\s*\[\s*Subtask\s*(\d+)\s*(?:-\s*([^\]\s]+))?\s*\]\s*")

    def __init__(self, dataset_dir: Path, stn: int, validator: Path | None) -> None:
        self._dir = Path(dataset_dir, f"st{stn}")
        self.commands: list[_Command] = []
        self.extends: list[_Extends] = []
        self.validator = validator
        self.parents: set[Stn] = set()

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

    def __init__(self, group: _GroupName) -> None:
        self._group = group

    def dst_file(self, directory: Path, idx: int) -> Path:
        return Path(directory, f"{self._group}-{idx}.in")

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

    @utils.work("copy", "{0}")
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

    @utils.work("echo", "{0}")
    def run(self, dst_dir: Path, tests_in_group: Counter[_GroupName]) -> utils.Result:
        idx = _next_idx_in_group(self._group, tests_in_group)
        with self.dst_file(dst_dir, idx).open("w") as test_file:
            test_file.write(" ".join(self._args) + "\n")
            return _success_with_count_result(1)


class _Script(_Command):
    VALID_EXTENSIONS = Literal[".py", ".cpp"]

    def __init__(
        self,
        group: _GroupName,
        path: Path,
        ext: VALID_EXTENSIONS,
        args: list[str],
        cwd: Path,
    ) -> None:
        super().__init__(group)
        self._cwd = cwd
        self._args = args
        self._script_path = path
        self._ext = ext

    def __str__(self) -> str:
        args = " ".join(self._args)
        script = self._script_path.name
        return f"{script} {args}"

    @utils.work("gen", "{0}")
    def run(self, dst_dir: Path, tests_in_group: Counter[_GroupName]) -> utils.Result:
        script = self._load_script()
        if not script:
            return utils.Result.fail(short_msg="script file not found")
        build_result = script.build()
        if isinstance(build_result, BuildError):
            return utils.Result.fail(
                short_msg="failed to build generator",
                long_msg=build_result.msg,
            )

        count = 0
        # We seed the script with the next `idx`, this guarantees it will be different next time.
        args = self._args_with_seed(dst_dir, tests_in_group[self._group] + 1)
        process = build_result.spawn(args, cwd=self._cwd)
        if isinstance(process, Error):
            return utils.Result.fail(
                short_msg="error when running script",
                long_msg=process.msg,
            )
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
                        idx = _next_idx_in_group(self._group, tests_in_group)
                        current_file = self.dst_file(dst_dir, idx).open("w")
                    current_file.write(char)
        finally:
            if current_file:
                current_file.close()

        ret = process.wait()
        if ret != 0:
            msg = ret_code_to_str(ret)
            args_fmt = " ".join(args)
            script_path = utils.relative_to_cwd(self._script_path)
            cmd = f"$ {script_path} {args_fmt}"
            long_msg = f"{cmd}\n{process.stderr.read()}" if process.stderr else cmd
            return utils.Result.fail(short_msg=msg, long_msg=long_msg)

        if count == 0:
            return utils.Result.fail(short_msg="generator didn't produce any output")

        return _success_with_count_result(count)

    def _load_script(self) -> SourceCode | None:
        if not self._script_path.exists():
            return None
        if self._ext == ".py":
            return PythonSource(self._script_path)
        elif self._ext == ".cpp":
            return CppSource(self._script_path)

    def _args_with_seed(self, directory: Path, idx: int) -> list[str]:
        return [f"{directory.name}-{self._group}-{idx}", *self._args]


def _invalid_command_err_msg(cmd: str) -> str:
    extensions = typing.get_args(_Script.VALID_EXTENSIONS)
    return (
        f"invalid command `{cmd}`\n"
        f"The command should be either `copy`, `echo` or a generator with one of the following extensions {extensions}"
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


def _has_cycles(subtasks: SortedDict[Stn, _Subtask]) -> bool:
    visited: set[Stn] = set()
    stack: set[Stn] = set()

    def dfs(sti: Stn) -> bool:
        visited.add(sti)
        stack.add(sti)

        for extends in subtasks[sti].extends:
            if extends.stn not in visited:
                if dfs(extends.stn):
                    return True
            elif extends.stn in stack:
                return True

        stack.remove(sti)
        return False

    return any(dfs(node) for node in subtasks)
