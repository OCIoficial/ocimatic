from __future__ import annotations

from enum import IntEnum
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

        parser = Parser()
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


class _TokenKind(IntEnum):
    OpenBracket = 0
    CloseBracket = 1
    Directive = 2
    Word = 3
    String = 4
    Int = 5
    Eol = 6
    Error = 7

    def __str__(self) -> str:
        match self:
            case _TokenKind.OpenBracket:
                return "["
            case _TokenKind.CloseBracket:
                return "]"
            case _TokenKind.Directive:
                return "directive"
            case _TokenKind.Word:
                return "word"
            case _TokenKind.String:
                return "string"
            case _TokenKind.Int:
                return "int"
            case _TokenKind.Eol:
                return "end of line"
            case _TokenKind.Error:
                return "error"


@dataclass(kw_only=True, frozen=True, slots=True)
class _Token:
    range: Range
    lexeme: str
    kind: _TokenKind


type _Peek = _TokenKind | set[_TokenKind] | str


class _Scanner:
    COMMENT_RE = re.compile(r"\s*(#.*)?")
    HEADER_RE = re.compile(r"\[\s*Subtask\s+(\d+)\s*\]")
    STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
    EXTENDS_RE = re.compile(r"@extends\s*subtask\s*(\d+)")
    VALIDATOR_RE = re.compile(r"@validator\s*(\S+)")

    DIRECTIVE_RE = re.compile(r"@[a-z]+")
    INT_RE = re.compile(r"\d+")
    WORD_RE = re.compile(r"[a-zA-Z0-9_\./*-]+")

    def __init__(self, lineno: int, line: str) -> None:
        self._lineno = lineno
        self._pos = 0
        self._line = line
        self._hi: Position = Position(line=lineno, column=0)
        self._scan()

    def _scan(self) -> None:
        if m := _Scanner.COMMENT_RE.match(self._line, pos=self._pos):
            self._pos = m.end(0)

        if self._pos == len(self._line):
            kind, span = (_TokenKind.Eol, (self._pos, self._pos + 1))
        elif self._line[self._pos] == "[":
            kind, span = (_TokenKind.OpenBracket, (self._pos, self._pos + 1))
        elif self._line[self._pos] == "]":
            kind, span = (_TokenKind.CloseBracket, (self._pos, self._pos + 1))
        elif m := _Scanner.DIRECTIVE_RE.match(self._line, pos=self._pos):
            kind, span = (_TokenKind.Directive, m.span(0))
        elif m := _Scanner.INT_RE.match(self._line, pos=self._pos):
            kind, span = (_TokenKind.Int, m.span(0))
        elif m := _Scanner.WORD_RE.match(self._line, pos=self._pos):
            kind, span = (_TokenKind.Word, m.span(0))
        elif m := _Scanner.STRING_RE.match(self._line, pos=self._pos):
            kind, span = (_TokenKind.String, m.span(0))
        else:
            kind, span = (_TokenKind.Error, (self._pos, self._pos + 1))
        self._pos = span[1]
        self._next_token = _Token(
            kind=kind,
            lexeme=self._line[span[0] : span[1]],
            range=Range(
                start=Position(line=self._lineno, column=span[0]),
                end=Position(line=self._lineno, column=span[1]),
            ),
        )

    def is_eol(self) -> bool:
        return self._next_token.kind == _TokenKind.Eol

    def peek(self, peek: _Peek) -> bool:
        if isinstance(peek, str):
            return self._next_token.lexeme == peek
        elif isinstance(peek, set):
            return any(self._next_token.kind == k for k in peek)
        else:
            return self._next_token.kind == peek

    def next_if(self, p: _Peek) -> _Token | None:
        if self.peek(p):
            return self.next()

    def next(self) -> _Token:
        token = self._next_token
        self._hi = token.range.end
        self._scan()
        return token

    def expect(self, peek: _Peek, expected: set[str] | None = None) -> _Token:
        if self.peek(peek):
            return self.next()
        else:
            raise self.unexpected_token(expected or self._peek_to_expected(peek))

    @staticmethod
    def _peek_to_expected(peek: _Peek) -> set[str]:
        if isinstance(peek, str):
            return {peek}
        elif isinstance(peek, set):
            return {str(k) for k in peek}
        else:
            return {str(peek)}

    def pos(self) -> Position:
        """Return the start position of the next token."""
        return self._next_token.range.start

    def last_pos(self) -> Position:
        """Return the end position of the previously yielded token."""
        return self._hi

    def unexpected_token(self, expected: set[str] | None = None) -> ParseError:
        if expected:
            if len(expected) == 1:
                msg = f"expected '{next(iter(expected))}'"
            else:
                msg = f"expected one of {expected}"
        elif self.is_eol():
            msg = "unexpected end of line"
        else:
            msg = f"unexpected token `{self._next_token.lexeme}`"
        return ParseError(msg=msg, range=self._next_token.range)


class Parser:
    def __init__(self) -> None:
        self.subtasks: list[tuple[_SubtaskHeader, list[_Item]]] = []
        self.errors: list[ParseError] = []

    def parse(self, content: str) -> None:
        for lineno, line in enumerate(content.splitlines()):
            scanner = _Scanner(lineno, line)

            # Skip empty lines
            if scanner.is_eol():
                continue

            try:
                parsed = self._parse_line(scanner)
                scanner.expect(_TokenKind.Eol)
            except ParseError as err:
                self.errors.append(err)
                continue

            match parsed:
                case _SubtaskHeader() as header:
                    self.subtasks.append((header, []))
                case item:
                    if self.subtasks:
                        self.subtasks[-1][1].append(item)
                    else:
                        self.errors.append(
                            ParseError(
                                msg="unexpected item before first subtask",
                                range=item.range,
                            ),
                        )

    def _parse_line(self, scanner: _Scanner) -> _SubtaskHeader | _Item:
        if scanner.peek(_TokenKind.OpenBracket):
            return self._parse_header(scanner)
        elif scanner.peek(_TokenKind.Directive):
            return self._parse_directive(scanner)
        elif scanner.peek(_TokenKind.Word):
            return self._parse_command(scanner)
        else:
            raise scanner.unexpected_token({"header", "directive", "command"})

    def _parse_header(self, scanner: _Scanner) -> _SubtaskHeader:
        start = scanner.pos()
        scanner.expect(_TokenKind.OpenBracket)
        scanner.expect("Subtask")
        num = scanner.expect(_TokenKind.Int)
        scanner.expect(_TokenKind.CloseBracket)
        end = scanner.last_pos()

        return _SubtaskHeader(number=int(num.lexeme), range=Range(start=start, end=end))

    def _parse_directive(self, scanner: _Scanner) -> _Extends | _Validator:
        if scanner.peek("@extends"):
            return self._parse_extends(scanner)
        elif scanner.peek("@validator"):
            return self._parse_validator(scanner)
        else:
            raise scanner.unexpected_token({"@extends", "@validator"})

    def _parse_extends(self, scanner: _Scanner) -> _Extends:
        start = scanner.pos()
        scanner.expect("@extends")
        scanner.expect("subtask")
        num = scanner.expect(_TokenKind.Int)
        end = scanner.last_pos()
        return _Extends(stn=Stn(int(num.lexeme)), range=Range(start=start, end=end))

    def _parse_validator(self, scanner: _Scanner) -> _Validator:
        start = scanner.pos()
        scanner.expect("@validator")
        path = scanner.expect(_TokenKind.Word)
        end = scanner.last_pos()

        return _Validator(path=path.lexeme, range=Range(start=start, end=end))

    def _parse_command(self, scanner: _Scanner) -> _Command:
        start = scanner.pos()
        group = self._validate_group_name(scanner.next())
        scanner.expect(";")
        cmd_start = scanner.pos()
        cmd = scanner.expect(_TokenKind.Word, {"copy", "echo", "script"})
        args = self._parse_command_args(scanner)
        end = scanner.last_pos()

        range = Range(start=start, end=end)
        if cmd.lexeme == "copy":
            if len(args) != 1:
                raise ParseError(
                    msg="the `copy` command expects exactly one argument.",
                    range=Range(start=cmd_start, end=end),
                )
            return _Copy(group, args[0], range)
        elif cmd.lexeme == "echo":
            return _Echo(group, args, range)
        elif (ext := Path(cmd.lexeme).suffix) in (".py", ".cpp"):
            # mypy can't tell `ext` is either `.py` or `.cpp` from the check above
            return _Script(
                group,
                cmd.lexeme,
                cast(_Script.VALID_EXTENSIONS, ext),  # pyright: ignore [reportUnnecessaryCast]
                args,
                range,
            )
        else:
            raise _invalid_command_err(cmd)

    def _validate_group_name(self, group: _Token) -> _GroupName:
        if _GroupName.RE.fullmatch(group.lexeme) is None:
            raise ParseError(
                msg=f"invalid group name: `{group.lexeme}`\nGroup name must match the following regular expression: `{_GroupName.RE.pattern}`",
                range=group.range,
            )
        return _GroupName(group.lexeme)

    def _parse_command_args(self, scanner: _Scanner) -> list[str]:
        args: list[str] = []
        while not scanner.is_eol():
            if t := scanner.next_if(_TokenKind.String):
                args.append(t.lexeme.encode().decode("unicode_escape"))
            elif t := scanner.next_if({_TokenKind.Word, _TokenKind.Int}):
                args.append(t.lexeme)
            else:
                raise scanner.unexpected_token()
        return args


@dataclass(kw_only=True, frozen=True)
class ParseError(Exception):
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
        return f"{self.line + 1}:{self.column + 1}"


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


@dataclass(frozen=True, slots=True)
class _GroupName:
    RE = re.compile(r"[a-zA-Z0-9_-]+")

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
    def __init__(self, group: _GroupName, range: Range) -> None:
        self._group = group
        self.range = range

    @abstractmethod
    def run(self, cx: _CommandCtxt) -> Result: ...


class _Copy(_Command):
    magic_check = re.compile("([*?[])")

    def __init__(self, group: _GroupName, pattern: str, range: Range) -> None:
        super().__init__(group, range)
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
    def __init__(self, group: _GroupName, args: list[str], range: Range) -> None:
        super().__init__(group, range)
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
        range: Range,
    ) -> None:
        super().__init__(group, range)
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


def _invalid_command_err(cmd: _Token) -> ParseError:
    extensions = typing.get_args(_Script.VALID_EXTENSIONS)
    msg = (
        f"invalid command `{cmd.lexeme}`\n"
        f"The command should be either `copy`, `echo` or a generator script with one of the following extensions {extensions}"
    )
    return ParseError(msg=msg, range=cmd.range)


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
