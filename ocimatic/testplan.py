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
from typing import ClassVar, Literal, cast

from ocimatic import ui, utils
from ocimatic.result import Error, Result, Status
from ocimatic.runnable import ret_code_to_str
from ocimatic.source_code import BuildError, CppSource, PythonSource, SourceCode
from ocimatic.utils import SortedDict, Stn

# <https://en.wikipedia.org/wiki/C0_and_C1_control_codes#FS>
FS = chr(28)


class Testplan:
    """Functionality to read and run a plan for generating dataset."""

    COMMENT_RE = re.compile(r"([^#]*)(#.*)?", re.RegexFlag.DOTALL)

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
            ui.fatal_error(
                f'File not found: "{self._testplan_path}"',
            )
        self._task_directory = task_directory
        self._dataset_dir = dataset_directory

        parser = _Parser(self._testplan_path, task_directory)
        parser.parse()

        if len(parser.errors) > 0:
            ui.writeln(
                f"Error when parsing testplan in `{utils.relative_to_cwd(self._testplan_path)}`",
                ui.ERROR,
            )
            ui.writeln(f"{parser.errors[0]}", ui.ERROR)
            sys.exit(1)

        subtasks = self._validate_subtasks(parser.subtasks)
        if isinstance(subtasks, ParseError):
            ui.writeln(
                f"Error when parsing testplan in `{utils.relative_to_cwd(self._testplan_path)}`",
                ui.ERROR,
            )
            ui.writeln(f"{subtasks}", ui.ERROR)
            sys.exit(1)

        self._subtasks = subtasks

    @property
    def subtasks(self) -> int:
        return len(self._subtasks)

    def validators(self) -> SortedDict[Stn, Path | None]:
        return SortedDict(
            (sti, st.validator.path if st.validator else None)
            for sti, st in self._subtasks.items()
        )

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

    def run(self, stn: Stn | None) -> Status:
        status = Status.success
        for sti, st in self._subtasks.items():
            if stn is not None and stn != sti:
                continue
            status &= st.run(self._task_directory)

        if sum(len(st.commands) for st in self._subtasks.values()) == 0:
            ui.show_message(
                "Warning",
                "no commands were executed for the plan.",
                ui.WARNING,
            )

        return status

    def _validate_subtasks(
        self,
        parsed: list[tuple[_SubtaskHeader, list[Item]]],
    ) -> SortedDict[Stn, _Subtask] | ParseError:
        subtasks: SortedDict[Stn, _Subtask] = SortedDict()
        for i, (header, items) in enumerate(parsed, start=1):
            if i != header.number:
                return ParseError(
                    lineno=header.lineno,
                    msg=f"found {header}, but [Subtask {i}] was expected",
                )
            sti = Stn(i)

            validator = None
            for item in items:
                if not isinstance(item, _Validator):
                    continue
                if validator is not None:
                    return ParseError(
                        lineno=item.lineno,
                        msg="multiple @validator directives found for the same subtask.",
                    )
                validator = item

            commands = [item for item in items if isinstance(item, _Command)]
            extends = [item for item in items if isinstance(item, _Extends)]

            subtask = _Subtask(self._dataset_dir, commands, extends, validator, sti)

            subtasks[sti] = subtask

        error = Testplan._validate_extends_graph(subtasks)
        if error is not None:
            return error

        return subtasks

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


class _Parser:
    def __init__(self, path: Path, task_directory: Path) -> None:
        self.subtasks: list[tuple[_SubtaskHeader, list[Item]]] = []
        self.errors: list[ParseError] = []

        self._path = path

    def parse(self) -> None:
        header = None
        items: list[Item] = []
        for lineno, line in enumerate(self._path.open("r").readlines(), 1):
            # Remove comments
            m = Testplan.COMMENT_RE.fullmatch(line)
            if m is None:
                continue
            line = m.group(1)

            # Skip empty lines
            line = line.strip()
            if not line:
                continue

            if m := _SubtaskHeader.RE.fullmatch(line):
                if header is not None:
                    self.subtasks.append((header, items))

                number = int(m.group(1))
                header = _SubtaskHeader(number=number, lineno=lineno)
                items = []
                continue

            if header is None:
                self.append_error(lineno, "unexpected line before first subtask header")
                continue

            if m := _Command.RE.fullmatch(line):
                command = self._parse_command(m, lineno)
                if command is not None:
                    items.append(command)
            elif m := _Extends.RE.fullmatch(line):
                items.append(_Extends(stn=Stn(int(m.group(1))), lineno=lineno))
            elif m := _Validator.RE.fullmatch(line):
                path = Path(self._path.parent, m.group(1))
                items.append(_Validator(path=path, lineno=lineno))
            else:
                self.append_error(lineno, f"invalid line `{line}`")

        if header is not None:
            self.subtasks.append((header, items))

    def _parse_command(self, m: re.Match[str], lineno: int) -> _Command | None:
        group = _GroupName.parse(m.group(1))
        if isinstance(group, Error):
            self.append_error(lineno, group.msg)
            return None

        cmd = m.group(2)
        args = _parse_args(m.group(3) or "")

        if cmd == "copy":
            if len(args) > 2:
                self.append_error(
                    lineno,
                    "the `copy` command expects exactly one argument.",
                )
                return None
            return _Copy(group, args[0])
        elif cmd == "echo":
            return _Echo(group, args)
        elif (ext := Path(cmd).suffix) in (".py", ".cpp"):
            # mypy can't tell `ext` is either `.py` or `.cpp` from the check above
            return _Script(
                group,
                Path(self._path.parent, cmd),
                cast(_Script.VALID_EXTENSIONS, ext),  # pyright: ignore [reportUnnecessaryCast]
                args,
                self._path.parent,
            )
        else:
            self.append_error(lineno, _invalid_command_err_msg(cmd))
            return None

    def append_error(self, lineno: int, msg: str) -> None:
        self.errors.append(ParseError(lineno=lineno, msg=msg))


@dataclass(kw_only=True, frozen=True, slots=True)
class ParseError:
    lineno: int | None = None
    msg: str

    def __str__(self) -> str:
        if self.lineno:
            return f"line {self.lineno}: {self.msg}"
        else:
            return self.msg


type Item = _Validator | _Extends | _Command


@dataclass(kw_only=True, frozen=True, slots=True)
class _SubtaskHeader:
    RE: ClassVar[re.Pattern[str]] = re.compile(r"\s*\[\s*Subtask\s+(\d+)\s*\]\s*")

    number: int
    lineno: int

    def __str__(self) -> str:
        return f"[Subtask {self.number}]"


@dataclass(kw_only=True, frozen=True, slots=True)
class _Validator:
    """A validator directive can be used to define an input validator for a subtask."""

    RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\s*@\s*validator\s*(\S+)\s*",
    )

    path: Path
    lineno: int

    def __str__(self) -> str:
        return f"@validator {self.path}"


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
    def parse(name: str) -> _GroupName | Error:
        if not _GroupName.RE.fullmatch(name):
            return Error(
                msg="invalid group name `{group_str}`. The group name should "
                f"match the regex `{_GroupName.RE.pattern}`.",
            )
        return _GroupName(name=name)

    def __str__(self) -> str:
        return self.name


class _Subtask:
    def __init__(
        self,
        dataset_dir: Path,
        commands: list[_Command],
        extends: list[_Extends],
        validator: _Validator | None,
        stn: Stn,
    ) -> None:
        self._dir = Path(dataset_dir, f"st{stn}")
        self.extends = extends
        self.validator = validator
        self.commands = commands

    def __str__(self) -> str:
        return str(self._dir.name)

    @ui.hd2("{0}")
    def run(self, task_dir: Path) -> Status:
        shutil.rmtree(self._dir, ignore_errors=True)
        self._dir.mkdir(parents=True, exist_ok=True)

        status = Status.success
        tests_in_group: Counter[_GroupName] = Counter()
        for cmd in self.commands:
            status &= cmd.run(task_dir, self._dir, tests_in_group).status
        return status


class _Command(ABC):
    RE = re.compile(r"\s*([^;\s]+)\s*;\s*(\S+)(:?\s+(.*))?")

    def __init__(self, group: _GroupName) -> None:
        self._group = group

    def _next_idx_in_group(self, tests_in_group: Counter[_GroupName]) -> int:
        tests_in_group[self._group] += 1
        return tests_in_group[self._group]

    def dst_file(self, directory: Path, idx: int) -> Path:
        return Path(directory, f"{self._group}-{idx}.in")

    @abstractmethod
    def run(
        self,
        task_dir: Path,
        dst_dir: Path,
        tests_in_group: Counter[_GroupName],
    ) -> Result:
        raise NotImplementedError(
            f"Class {self.__class__.__name__} doesn't implement run()",
        )


class _Copy(_Command):
    magic_check = re.compile("([*?[])")

    def __init__(self, group: _GroupName, pattern: str) -> None:
        super().__init__(group)
        self._pattern = pattern

    def __str__(self) -> str:
        return self._pattern

    @ui.work("copy", "{0}")
    def run(
        self,
        task_dir: Path,
        dst_dir: Path,
        tests_in_group: Counter[_GroupName],
    ) -> Result:
        files = list(task_dir.glob(self._pattern))
        if not files:
            msg = "No file matches the pattern" if self.has_magic() else "No such file"
            return Result.fail(short_msg=msg)
        try:
            for file in files:
                idx = self._next_idx_in_group(tests_in_group)
                shutil.copy(file, self.dst_file(dst_dir, idx))
            return _success_with_count_result(len(files))
        except Exception:  # pylint: disable=broad-except
            return Result.fail(short_msg="Error when copying file")

    def has_magic(self) -> bool:
        return _Copy.magic_check.search(self._pattern) is not None


class _Echo(_Command):
    def __init__(self, group: _GroupName, args: list[str]) -> None:
        super().__init__(group)
        self._args = args

    def __str__(self) -> str:
        return str(self._args)

    @ui.work("echo", "{0}")
    def run(
        self,
        task_dir: Path,
        dst_dir: Path,
        tests_in_group: Counter[_GroupName],
    ) -> Result:
        del task_dir
        idx = self._next_idx_in_group(tests_in_group)
        with self.dst_file(dst_dir, idx).open("w") as test_file:
            test_file.write(" ".join(self._args) + "\n")
            return _success_with_count_result(1)


class _Script(_Command):
    VALID_EXTENSIONS = Literal[".py", ".cpp"]
    _ext: VALID_EXTENSIONS

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

    @ui.work("gen", "{0}")
    def run(
        self,
        task_dir: Path,
        dst_dir: Path,
        tests_in_group: Counter[_GroupName],
    ) -> Result:
        del task_dir
        script = self._load_script()
        if not script:
            return Result.fail(short_msg="script file not found")
        build_result = script.build()
        if isinstance(build_result, BuildError):
            return Result.fail(
                short_msg="failed to build generator",
                long_msg=build_result.msg,
            )

        count = 0
        # We seed the script with the next `idx`, this guarantees it will be different next time.
        args = self._args_with_seed(dst_dir, tests_in_group[self._group] + 1)
        process = build_result.spawn(args, cwd=self._cwd)
        if isinstance(process, Error):
            return Result.fail(
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
                        idx = self._next_idx_in_group(tests_in_group)
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
            return Result.fail(short_msg=msg, long_msg=long_msg)

        if count == 0:
            return Result.fail(short_msg="generator didn't produce any output")

        return _success_with_count_result(count)

    def _load_script(self) -> SourceCode | None:
        if not self._script_path.exists():
            return None
        match self._ext:
            case ".py":
                return PythonSource(self._script_path)
            case ".cpp":
                return CppSource(self._script_path)

    def _args_with_seed(self, directory: Path, idx: int) -> list[str]:
        return [f"{directory.name}-{self._group}-{idx}", *self._args]


def _invalid_command_err_msg(cmd: str) -> str:
    extensions = typing.get_args(_Script.VALID_EXTENSIONS)
    return (
        f"invalid command `{cmd}`\n"
        f"The command should be either `copy`, `echo` or a generator with one of the following extensions {extensions}"
    )


def _success_with_count_result(count: int) -> Result:
    assert count > 0
    if count == 1:
        return Result.success(short_msg="1 test case generated")
    else:
        return Result.success(short_msg=f"{count} test cases generated")


def _parse_args(args: str) -> list[str]:
    args = args.strip()
    return [a.encode().decode("unicode_escape") for a in shlex.split(args)]


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
