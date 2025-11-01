from __future__ import annotations

import re
import shutil
import sys
import typing
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ocimatic import ui, utils
from ocimatic.result import Error, Result, Status
from ocimatic.runnable import ret_code_to_str
from ocimatic.source_code import BuildError, CppSource, PythonSource, SourceCode
from ocimatic.utils import SortedDict, Stn

# <https://en.wikipedia.org/wiki/C0_and_C1_control_codes#FS>
FS = chr(28)


class Testplan:
    """Functionality to read and run a plan for generating dataset."""

    def __init__(
        self,
        path: Path,
        task_directory: Path,
        dataset_directory: Path,
    ) -> None:
        self._path = path
        if not self._path.exists():
            ui.fatal_error(f'File not found: "{self._path}"')
        self._task_directory = task_directory
        self._dataset_dir = dataset_directory

        parser = _Parser()
        parser.parse(self._path.read_text())

        err_msg = f"Error when parsing testplan: `{utils.relative_to_cwd(self._path)}`"
        if len(parser.errors) > 0:
            ui.writeln(err_msg, ui.ERROR)
            ui.writeln(f"{parser.errors[0]}", ui.ERROR)
            sys.exit(1)

        if isinstance(subtasks := self._validate_subtasks(parser.subtasks), ParseError):
            ui.writeln(err_msg, ui.ERROR)
            ui.writeln(f"{subtasks}", ui.ERROR)
            sys.exit(1)

        self._subtasks = subtasks

    @property
    def subtasks(self) -> int:
        return len(self._subtasks)

    def validators(self) -> SortedDict[Stn, Path | None]:
        basedir = self._path.parent
        return SortedDict(
            (sti, Path(basedir, st.validator.path) if st.validator else None)
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
            status &= st.run(self._path.parent, self._task_directory)

        if sum(len(st.commands) for st in self._subtasks.values()) == 0:
            ui.show_message(
                "Warning",
                "no commands were executed for the plan.",
                ui.WARNING,
            )

        return status

    def _validate_subtasks(
        self,
        parsed: list[tuple[_SubtaskHeader, list[_Item]]],
    ) -> SortedDict[Stn, _Subtask] | ParseError:
        subtasks: SortedDict[Stn, _Subtask] = SortedDict()
        for i, (header, items) in enumerate(parsed, start=1):
            if i != header.number:
                return ParseError(
                    range=header.range,
                    msg=f"found {header}, but [Subtask {i}] was expected",
                )
            sti = Stn(i)

            validator = None
            for item in items:
                if not isinstance(item, _Validator):
                    continue
                if validator is not None:
                    return ParseError(
                        range=validator.range,
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
                range = extends.range
                if extends.stn in seen:
                    return ParseError(
                        range=range,
                        msg=f"cannot extends twice from the same subtask: `{extends}`",
                    )
                if extends.stn not in subtasks:
                    return ParseError(
                        range=range,
                        msg=f"invalid subtask {extends.stn}: `{extends}`",
                    )
                if extends.stn == sti:
                    return ParseError(
                        range=range,
                        msg=f"a subtask cannot extend itself: `{subtasks}`",
                    )
                seen.add(extends.stn)

        if _has_cycles(subtasks):
            return ParseError(msg="the extends graph contains cycles")

        return None


class _Scanner:
    COMMENT_RE = re.compile(r"\s*(#.*)?")
    HEADER_RE = re.compile(r"\[\s*Subtask\s+(\d+)\s*\]")
    WORD_RE = re.compile(r"[a-zA-Z0-9_\./*-]+")
    STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
    GROUP_RE = re.compile(r"[a-zA-Z0-9_-]+")
    EXTENDS_RE = re.compile(r"@extends\s*subtask\s*(\d+)")
    VALIDATOR_RE = re.compile(r"@validator\s*(\S+)")

    def __init__(self, lineno: int, line: str) -> None:
        self._lineno = lineno
        self._pos = 0
        self._line = line
        self._skip_comment()

    def is_eol(self) -> bool:
        if self._pos == len(self._line):
            return True
        return self._line[self._pos] == "\n"

    def scan(self, pattern: re.Pattern[str]) -> re.Match[str] | None:
        if (m := pattern.match(self._line, pos=self._pos)) is not None:
            self._pos = m.end(0)
            self._skip_comment()

        return m

    def range(self, m: re.Match[str]) -> Range:
        start = Position(line=self._lineno, column=m.start(0))
        end = Position(line=self._lineno, column=m.end(0))
        return Range(start=start, end=end)

    def line_range(self) -> Range:
        start = Position(line=self._lineno, column=0)
        end = Position(line=self._lineno, column=len(self._line))
        return Range(start=start, end=end)

    def expect(self, c: str) -> bool:
        if self._pos < len(self._line) and self._line[self._pos] == c:
            self._pos += 1
            self._skip_comment()
            return True
        else:
            return False

    def unexpected_token(self) -> ParseError:
        if self._pos == len(self._line):
            msg = "unexpected end of line"
        else:
            msg = f"unexpected token `{self._line[self._pos]}`"
        start = Position(line=self._lineno, column=self._pos)
        end = Position(line=self._lineno, column=self._pos + 1)
        return ParseError(msg=msg, range=Range(start=start, end=end))

    def _skip_comment(self) -> None:
        self._scan_inner(_Scanner.COMMENT_RE)

    def _scan_inner(self, pattern: re.Pattern[str]) -> re.Match[str] | None:
        if (m := pattern.match(self._line, pos=self._pos)) is not None:
            self._pos = m.end(0)
        return m


class _Parser:
    def __init__(self) -> None:
        self.subtasks: list[tuple[_SubtaskHeader, list[_Item]]] = []
        self.errors: list[ParseError] = []

    def parse(self, content: str) -> None:
        header = None
        items: list[_Item] = []
        for lineno, line in enumerate(content.splitlines(), 1):
            scanner = _Scanner(lineno, line)

            if m := scanner.scan(_Scanner.HEADER_RE):
                if header is not None:
                    self.subtasks.append((header, items))

                number = int(m.group(1))
                header = _SubtaskHeader(number=number, range=scanner.range(m))
                items = []
            elif header is None:
                self.append_error(
                    scanner.line_range(),
                    "unexpected line before first subtask header",
                )
                continue
            elif (item := self._parse_item(scanner)) is not None:
                items.append(item)
            self._expect_eol(scanner)

        if header is not None:
            self.subtasks.append((header, items))

    def _parse_item(self, scanner: _Scanner) -> _Item | None:
        if m := scanner.scan(_Scanner.GROUP_RE):
            group = _GroupName(name=m.group(0))
            return self._parse_command(group, scanner)
        elif m := scanner.scan(_Scanner.EXTENDS_RE):
            return _Extends(stn=Stn(int(m.group(1))), range=scanner.range(m))
        elif m := scanner.scan(_Scanner.VALIDATOR_RE):
            return _Validator(path=m.group(1), range=scanner.range(m))
        else:
            return None

    def _parse_command(self, group: _GroupName, scanner: _Scanner) -> _Command | None:
        if not scanner.expect(";"):
            self.errors.append(scanner.unexpected_token())
            return None

        if (m := scanner.scan(_Scanner.WORD_RE)) is None:
            self.errors.append(scanner.unexpected_token())
            return None
        cmd = m.group(0)

        if (args := self._parse_command_args(scanner)) is None:
            return None

        if cmd == "copy":
            if len(args) > 2:
                self.append_error(
                    scanner.line_range(),
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
                cmd,
                cast(_Script.VALID_EXTENSIONS, ext),  # pyright: ignore [reportUnnecessaryCast]
                args,
            )
        else:
            self.append_error(scanner.range(m), _invalid_command_err_msg(cmd))
            return None

    def _parse_command_args(self, scanner: _Scanner) -> list[str] | None:
        args: list[str] = []
        while not scanner.is_eol():
            if m := scanner.scan(_Scanner.STRING_RE):
                args.append(m.group(1).encode().decode("unicode_escape"))
            elif m := scanner.scan(_Scanner.WORD_RE):
                args.append(m.group(0))
            else:
                self.errors.append(scanner.unexpected_token())
                return None
        return args

    def _expect_eol(self, scanner: _Scanner) -> None:
        if not scanner.is_eol():
            self.errors.append(scanner.unexpected_token())

    def append_error(self, range: Range, msg: str) -> None:
        self.errors.append(ParseError(range=range, msg=msg))


@dataclass(kw_only=True, frozen=True, slots=True)
class ParseError:
    range: Range | None = None
    msg: str

    def __str__(self) -> str:
        if self.range:
            return f"{self.range}: {self.msg}"
        else:
            return self.msg


type _Item = _Validator | _Extends | _Command


@dataclass(kw_only=True, frozen=True, slots=True)
class Position:
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.line}:{self.column}"


@dataclass(kw_only=True, frozen=True, slots=True)
class Range:
    start: Position
    end: Position

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"


@dataclass(kw_only=True, frozen=True, slots=True)
class _SubtaskHeader:
    number: int
    range: Range

    def __str__(self) -> str:
        return f"[Subtask {self.number}]"


@dataclass(kw_only=True, frozen=True, slots=True)
class _Validator:
    """A validator directive can be used to define an input validator for a subtask."""

    path: str
    range: Range

    def __str__(self) -> str:
        return f"@validator {self.path}"


@dataclass(kw_only=True, frozen=True, slots=True)
class _Extends:
    """An extends directive can be used to include all tests from another subtask."""

    stn: Stn
    range: Range

    def __str__(self) -> str:
        return f"@extends subtask {self.stn}"


@dataclass(kw_only=True, frozen=True, slots=True)
class _GroupName:
    name: str

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
        self._dst_dir = Path(dataset_dir, f"st{stn}")
        self.extends = extends
        self.validator = validator
        self.commands = commands

    def __str__(self) -> str:
        return str(self._dst_dir.name)

    @ui.hd2("{0}")
    def run(self, cwd: Path, task_dir: Path) -> Status:
        shutil.rmtree(self._dst_dir, ignore_errors=True)
        self._dst_dir.mkdir(parents=True, exist_ok=True)

        cx = _CommandCtxt(
            cwd=cwd,
            task_dir=task_dir,
            tests_in_group=Counter(),
            dst_dir=self._dst_dir,
        )
        status = Status.success
        for cmd in self.commands:
            status &= cmd.run(cx).status
        return status


@dataclass(kw_only=True)
class _CommandCtxt:
    """Context used to execute commands for a single subtask."""

    cwd: Path
    task_dir: Path
    dst_dir: Path
    tests_in_group: Counter[_GroupName]

    def next_file(self, group: _GroupName) -> Path:
        self.tests_in_group[group] += 1
        idx = self.tests_in_group[group]
        return Path(self.dst_dir, f"{group}-{idx}.in")

    def script_path(self, filename: str) -> Path:
        return self.cwd / filename

    def load_script(self, filename: str, ext: _Script.VALID_EXTENSIONS) -> SourceCode:
        path = self.script_path(filename)
        match ext:
            case ".py":
                return PythonSource(path)
            case ".cpp":
                return CppSource(path)


class _Command(ABC):
    def __init__(self, group: _GroupName) -> None:
        self._group = group

    @abstractmethod
    def run(self, cx: _CommandCtxt) -> Result: ...


class _Copy(_Command):
    magic_check = re.compile("([*?[])")

    def __init__(self, group: _GroupName, pattern: str) -> None:
        super().__init__(group)
        self._pattern = pattern

    def __str__(self) -> str:
        return self._pattern

    @ui.work("copy", "{0}")
    def run(self, cx: _CommandCtxt) -> Result:
        files = list(cx.task_dir.glob(self._pattern))
        if not files:
            msg = "No file matches the pattern" if self.has_magic() else "No such file"
            return Result.fail(short_msg=msg)
        try:
            for file in files:
                shutil.copy(file, cx.next_file(self._group))
            return _success_with_count_result(len(files))
        except Exception as e:
            return Result.fail(short_msg="Error when copying file", long_msg=str(e))

    def has_magic(self) -> bool:
        return _Copy.magic_check.search(self._pattern) is not None


class _Echo(_Command):
    def __init__(self, group: _GroupName, args: list[str]) -> None:
        super().__init__(group)
        self._args = args

    def __str__(self) -> str:
        return str(self._args)

    @ui.work("echo", "{0}")
    def run(self, cx: _CommandCtxt) -> Result:
        with cx.next_file(self._group).open("w") as test_file:
            test_file.write(" ".join(self._args) + "\n")
            return _success_with_count_result(1)


class _Script(_Command):
    VALID_EXTENSIONS = Literal[".py", ".cpp"]
    _ext: VALID_EXTENSIONS

    def __init__(
        self,
        group: _GroupName,
        filename: str,
        ext: VALID_EXTENSIONS,
        args: list[str],
    ) -> None:
        super().__init__(group)
        self._filename = filename
        self._args = args
        self._ext = ext

    def __str__(self) -> str:
        args = " ".join(self._args)
        script = self._filename
        return f"{script} {args}"

    @ui.work("gen", "{0}")
    def run(self, cx: _CommandCtxt) -> Result:
        script = cx.load_script(self._filename, self._ext)
        if isinstance(runnable := script.build(), BuildError):
            return Result.fail(
                short_msg="failed to build generator",
                long_msg=runnable.msg,
            )

        count = 0
        args = self._args_with_seed(cx)
        if isinstance(process := runnable.spawn(args, cwd=cx.cwd), Error):
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
                        current_file = cx.next_file(self._group).open("w")
                    current_file.write(char)
        finally:
            if current_file:
                current_file.close()

        ret = process.wait()
        if ret != 0:
            msg = ret_code_to_str(ret)
            args_fmt = " ".join(args)
            script_path = utils.relative_to_cwd(cx.script_path(self._filename))
            cmd = f"$ {script_path} {args_fmt}"
            long_msg = f"{cmd}\n{process.stderr.read()}" if process.stderr else cmd
            return Result.fail(short_msg=msg, long_msg=long_msg)

        if count == 0:
            return Result.fail(short_msg="generator didn't produce any output")

        return _success_with_count_result(count)

    def _args_with_seed(self, cx: _CommandCtxt) -> list[str]:
        # We seed the script with the next `idx`, this guarantees it is different
        # every time, even if the script generates more than one file.
        idx = cx.tests_in_group[self._group] + 1
        return [f"{cx.dst_dir.name}-{self._group}-{idx}", *self._args]


def _invalid_command_err_msg(cmd: str) -> str:
    extensions = typing.get_args(_Script.VALID_EXTENSIONS)
    return (
        f"invalid command `{cmd}`\n"
        f"The command should be either `copy`, `echo` or a generator script with one of the following extensions {extensions}"
    )


def _success_with_count_result(count: int) -> Result:
    assert count > 0
    if count == 1:
        return Result.success(short_msg="1 test case generated")
    else:
        return Result.success(short_msg=f"{count} test cases generated")


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
