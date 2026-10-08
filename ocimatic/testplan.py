from __future__ import annotations

import re
import shutil
import typing
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import ClassVar, Literal

from ocimatic import ui, utils
from ocimatic.errors import OcimaticError
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
            raise OcimaticError(f'File not found: "{self._path}"')
        self._task_directory = task_directory
        self._dataset_dir = dataset_directory

        parser = Parser()
        parser.parse(self._path.read_text())

        err_msg = f"Error when parsing testplan: `{utils.relative_to_cwd(self._path)}`"
        if len(parser.errors) > 0:
            raise OcimaticError(err_msg, details="\n".join(map(str, parser.errors)))

        if errors := validate(parser.subtasks):
            raise OcimaticError(err_msg, details="\n".join(map(str, errors)))

        self._subtasks = self._build_subtasks(parser.subtasks)

    @property
    def subtasks(self) -> int:
        return len(self._subtasks)

    def validators(self) -> SortedDict[Stn, Path | None]:
        basedir = self._path.parent
        return SortedDict(
            (sti, Path(basedir, st.validator.path.lexeme) if st.validator else None)
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

    def _build_subtasks(
        self,
        parsed: list[tuple[SubtaskHeader, list[Item]]],
    ) -> SortedDict[Stn, _Subtask]:
        """Build the subtasks of a testplan that `validate` accepted."""
        subtasks: SortedDict[Stn, _Subtask] = SortedDict()
        for i, (_, items) in enumerate(parsed, start=1):
            sti = Stn(i)
            validator = next(
                (item for item in items if isinstance(item, Validator)),
                None,
            )
            commands = [item for item in items if isinstance(item, Command)]
            extends = [item for item in items if isinstance(item, Extends)]
            subtasks[sti] = _Subtask(
                self._dataset_dir,
                commands,
                extends,
                validator,
                sti,
            )
        return subtasks


class TokenKind(IntEnum):
    OpenBracket = 0
    CloseBracket = 1
    Directive = 2
    Word = 3
    String = 4
    Num = 5
    Eol = 6
    Error = 7

    def describe(self) -> str:
        """Describe the kind for error messages.

        Kinds that match exact text are shown in backticks; the rest are plain descriptions.
        """
        match self:
            case TokenKind.OpenBracket:
                return "`[`"
            case TokenKind.CloseBracket:
                return "`]`"
            case TokenKind.Directive:
                return "a directive"
            case TokenKind.Word:
                return "a word"
            case TokenKind.String:
                return "a string"
            case TokenKind.Num:
                return "a number"
            case TokenKind.Eol:
                return "end of line"
            case TokenKind.Error:
                return "an error"


@dataclass(kw_only=True, frozen=True, slots=True)
class Token:
    range: Range
    lexeme: str
    kind: TokenKind


type _Peek = TokenKind | list[TokenKind] | str


class _Scanner:
    COMMENT_RE = re.compile(r"\s*(#.*)?")
    STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
    DIRECTIVE_RE = re.compile(r"@[a-z]+")
    WORD_RE = re.compile(r"[a-zA-Z0-9_\./*-]+")

    def __init__(self, lineno: int, line: str) -> None:
        self._lineno = lineno
        self._pos = 0
        self._line = line
        self._hi: Position = Position(line=lineno, column=0)
        self._advance()

    def _advance(self) -> None:
        if m := _Scanner.COMMENT_RE.match(self._line, pos=self._pos):
            self._pos = m.end(0)

        if self._pos == len(self._line):
            kind, span = (TokenKind.Eol, (self._pos, self._pos + 1))
        elif self._line[self._pos] == "[":
            kind, span = (TokenKind.OpenBracket, (self._pos, self._pos + 1))
        elif self._line[self._pos] == "]":
            kind, span = (TokenKind.CloseBracket, (self._pos, self._pos + 1))
        elif m := _Scanner.DIRECTIVE_RE.match(self._line, pos=self._pos):
            kind, span = (TokenKind.Directive, m.span(0))
        elif m := _Scanner.WORD_RE.match(self._line, pos=self._pos):
            if m.group(0).isnumeric():
                kind, span = (TokenKind.Num, m.span(0))
            else:
                kind, span = (TokenKind.Word, m.span(0))
        elif m := _Scanner.STRING_RE.match(self._line, pos=self._pos):
            kind, span = (TokenKind.String, m.span(0))
        else:
            kind, span = (TokenKind.Error, (self._pos, self._pos + 1))
        self._pos = span[1]
        self._next_token = Token(
            kind=kind,
            lexeme=self._line[span[0] : span[1]],
            range=Range(
                start=Position(line=self._lineno, column=span[0]),
                end=Position(line=self._lineno, column=span[1]),
            ),
        )

    def is_eol(self) -> bool:
        return self._next_token.kind == TokenKind.Eol

    def peek(self, peek: _Peek) -> bool:
        if isinstance(peek, str):
            return self._next_token.lexeme == peek
        elif isinstance(peek, list):
            return any(self._next_token.kind == k for k in peek)
        else:
            return self._next_token.kind == peek

    def next_if(self, p: _Peek) -> Token | None:
        if self.peek(p):
            return self.next()
        else:
            return None

    def next(self) -> Token:
        token = self._next_token
        self._hi = token.range.end
        self._advance()
        return token

    def expect(self, peek: _Peek, expected: list[str] | None = None) -> Token:
        if self.peek(peek):
            return self.next()
        else:
            raise self.unexpected_token(expected or self._peek_to_expected(peek))

    @staticmethod
    def _peek_to_expected(peek: _Peek) -> list[str]:
        if isinstance(peek, str):
            return [f"`{peek}`"]
        elif isinstance(peek, list):
            return [k.describe() for k in peek]
        else:
            return [peek.describe()]

    def pos(self) -> Position:
        """Return the start position of the next token."""
        return self._next_token.range.start

    def last_pos(self) -> Position:
        """Return the end position of the previously yielded token."""
        return self._hi

    def unexpected_token(self, expected: list[str] | None = None) -> ParseError:
        """Return an error for the next token.

        Items in `expected` are used as given: exact text should be in backticks and
        descriptions (e.g. "a path") should not.
        """
        if expected:
            if len(expected) == 1:
                msg = f"expected {expected[0]}"
            else:
                msg = f"expected {', '.join(expected[:-1])} or {expected[-1]}"
        elif self.is_eol():
            msg = "unexpected end of line"
        else:
            msg = f"unexpected token `{self._next_token.lexeme}`"
        return ParseError(msg=msg, range=self._next_token.range)


class Parser:
    # Escape sequences allowed inside strings and the characters they stand for. In
    # addition, `\xHH` stands for the character with code point HH (two hex digits).
    _ESCAPES: ClassVar[dict[str, str]] = {'"': '"', "\\": "\\", "n": "\n", "t": "\t"}
    _ESCAPE_RE = re.compile(r"\\(?:x([0-9a-fA-F]{2})|(.))")

    def __init__(self) -> None:
        self.subtasks: list[tuple[SubtaskHeader, list[Item]]] = []
        self.errors: list[ParseError] = []

    def parse(self, content: str) -> None:
        for lineno, line in enumerate(content.splitlines()):
            scanner = _Scanner(lineno, line)

            # Skip empty lines
            if scanner.is_eol():
                continue

            try:
                parsed = self._parse_line(scanner)
                scanner.expect(TokenKind.Eol)
            except ParseError as err:
                self.errors.append(err)
                continue

            match parsed:
                case SubtaskHeader() as header:
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

    def _parse_line(self, scanner: _Scanner) -> SubtaskHeader | Item:
        if scanner.peek(TokenKind.OpenBracket):
            return self._parse_header(scanner)
        elif scanner.peek(TokenKind.Directive):
            return self._parse_directive(scanner)
        elif scanner.peek(TokenKind.Word):
            return self._parse_command(scanner)
        else:
            raise scanner.unexpected_token()

    def _parse_header(self, scanner: _Scanner) -> SubtaskHeader:
        start = scanner.pos()
        scanner.expect(TokenKind.OpenBracket)
        scanner.expect("Subtask")
        num = scanner.expect(TokenKind.Num)
        scanner.expect(TokenKind.CloseBracket)
        end = scanner.last_pos()

        return SubtaskHeader(number=int(num.lexeme), range=Range(start=start, end=end))

    def _parse_directive(self, scanner: _Scanner) -> Extends | Validator:
        if scanner.peek("@extends"):
            return self._parse_extends(scanner)
        elif scanner.peek("@validator"):
            return self._parse_validator(scanner)
        else:
            raise scanner.unexpected_token(["`@extends`", "`@validator`"])

    def _parse_extends(self, scanner: _Scanner) -> Extends:
        start = scanner.pos()
        scanner.expect("@extends")
        scanner.expect("subtask")
        num = scanner.expect(TokenKind.Num)
        end = scanner.last_pos()
        n = int(num.lexeme)
        if n < 1:
            raise ParseError(
                msg="subtask number must be greater than or equal to 1",
                range=num.range,
            )
        return Extends(stn=Stn(n), range=Range(start=start, end=end))

    def _parse_validator(self, scanner: _Scanner) -> Validator:
        start = scanner.pos()
        scanner.expect("@validator")
        path = scanner.expect(TokenKind.Word, ["a path"])
        end = scanner.last_pos()

        return Validator(path=path, range=Range(start=start, end=end))

    def _parse_command(self, scanner: _Scanner) -> Command:
        start = scanner.pos()
        group = self._validate_group_name(scanner.next())
        scanner.expect(";")
        cmd_start = scanner.pos()
        cmd = scanner.expect(
            TokenKind.Word,
            ["`copy`", "`echo`", "a generator script"],
        )
        args = self._parse_args(scanner)
        end = scanner.last_pos()

        range = Range(start=start, end=end)
        if cmd.lexeme == "copy":
            if len(args) != 1:
                raise ParseError(
                    msg="the `copy` command expects exactly one argument.",
                    range=Range(start=cmd_start, end=end),
                )
            return Copy(group, range, args[0])
        elif cmd.lexeme == "echo":
            return Echo(group, range, args)
        elif Path(cmd.lexeme).suffix in (".py", ".cpp"):
            return Script(group, range, cmd, args)
        else:
            raise _invalid_command_err(cmd)

    def _validate_group_name(self, group: Token) -> GroupName:
        if GroupName.RE.fullmatch(group.lexeme) is None:
            raise ParseError(
                msg=f"invalid group name: `{group.lexeme}`\nGroup name must match the following regular expression: `{GroupName.RE.pattern}`",
                range=group.range,
            )
        return GroupName(group.lexeme)

    def _parse_args(self, scanner: _Scanner) -> list[str]:
        args: list[str] = []
        while not scanner.is_eol():
            if t := scanner.next_if(TokenKind.String):
                args.append(self._parse_string(t))
            elif t := scanner.next_if([TokenKind.Word, TokenKind.Num]):
                args.append(t.lexeme)
            else:
                raise scanner.unexpected_token()
        return args

    def _parse_string(self, token: Token) -> str:
        """Return the content of a string token with its escape sequences decoded."""
        # The scanner guarantees the lexeme is delimited by quotes and that every
        # backslash inside is followed by another character.
        content = token.lexeme[1:-1]
        line = token.range.start.line
        offset = token.range.start.column + 1  # skip the opening quote

        def unescape(m: re.Match[str]) -> str:
            if (digits := m.group(1)) is not None:
                return chr(int(digits, 16))
            if (c := m.group(2)) in self._ESCAPES:
                return self._ESCAPES[c]
            col = offset + m.start()
            if c == "x":
                msg = "expected two hex digits after `\\x`"
            else:
                msg = f"invalid escape sequence `\\{c}`"
            raise ParseError(
                msg=msg,
                range=Range(
                    start=Position(line=line, column=col),
                    end=Position(line=line, column=col + 2),
                ),
            )

        return self._ESCAPE_RE.sub(unescape, content)


@dataclass(kw_only=True, frozen=True)
class SourceError(Exception):
    """An error in a testplan, optionally pointing to where it is in the source."""

    range: Range | None = None
    msg: str

    def __str__(self) -> str:
        if self.range:
            return f"{self.range}: {self.msg}"
        else:
            return self.msg


@dataclass(kw_only=True, frozen=True)
class ParseError(SourceError):
    """A syntax error, found while parsing a single line."""


@dataclass(kw_only=True, frozen=True)
class ValidationError(SourceError):
    """An error in a testplan that parses, found by `validate`."""


type Item = Validator | Extends | Command


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
class SubtaskHeader:
    number: int
    range: Range

    def __str__(self) -> str:
        return f"[Subtask {self.number}]"


@dataclass(kw_only=True, frozen=True, slots=True)
class Validator:
    """A validator directive can be used to define an input validator for a subtask."""

    path: Token
    range: Range

    def __str__(self) -> str:
        return f"@validator {self.path.lexeme}"


@dataclass(kw_only=True, frozen=True, slots=True)
class Extends:
    """An extends directive can be used to include all tests from another subtask."""

    stn: Stn
    range: Range

    def __str__(self) -> str:
        return f"@extends subtask {self.stn}"


@dataclass(frozen=True, slots=True)
class GroupName:
    RE = re.compile(r"[a-zA-Z0-9_-]+")

    name: str

    def __str__(self) -> str:
        return self.name


class _Subtask:
    def __init__(
        self,
        dataset_dir: Path,
        commands: list[Command],
        extends: list[Extends],
        validator: Validator | None,
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
    tests_in_group: Counter[GroupName]

    def next_file(self, group: GroupName) -> Path:
        self.tests_in_group[group] += 1
        idx = self.tests_in_group[group]
        return Path(self.dst_dir, f"{group}-{idx}.in")

    def script_path(self, filename: str) -> Path:
        return self.cwd / filename

    def load_script(self, filename: str) -> SourceCode:
        path = self.script_path(filename)
        match path.suffix:
            case ".py":
                return PythonSource(path)
            case ".cpp":
                return CppSource(path)
            case _:
                # This is validated during parsing
                raise ValueError(f"Unsupported file type: {path.suffix}")


@dataclass(frozen=True)
class Command(ABC):
    group: GroupName
    range: Range

    @abstractmethod
    def run(self, cx: _CommandCtxt) -> Result: ...


@dataclass(frozen=True)
class Copy(Command):
    magic_check = re.compile("([*?[])")

    group: GroupName
    pattern: str

    def __str__(self) -> str:
        return self.pattern

    @ui.work("copy", "{0}")
    def run(self, cx: _CommandCtxt) -> Result:
        files = sorted(cx.task_dir.glob(self.pattern))
        if not files:
            msg = "No file matches the pattern" if self.has_magic() else "No such file"
            return Result.fail(short_msg=msg)
        try:
            for file in files:
                shutil.copy(file, cx.next_file(self.group))
            return _success_with_count_result(len(files))
        except Exception as e:
            return Result.fail(short_msg="Error when copying file", long_msg=str(e))

    def has_magic(self) -> bool:
        return Copy.magic_check.search(self.pattern) is not None


@dataclass(frozen=True)
class Echo(Command):
    args: list[str]

    def __str__(self) -> str:
        return str(self.args)

    @ui.work("echo", "{0}")
    def run(self, cx: _CommandCtxt) -> Result:
        with cx.next_file(self.group).open("w") as test_file:
            test_file.write(" ".join(self.args) + "\n")
            return _success_with_count_result(1)


@dataclass(frozen=True)
class Script(Command):
    VALID_EXTENSIONS = Literal[".py", ".cpp"]

    cmd: Token
    args: list[str]

    def __str__(self) -> str:
        args = " ".join(self.args)
        script = self.cmd.lexeme
        return f"{script} {args}"

    @ui.work("gen", "{0}")
    def run(self, cx: _CommandCtxt) -> Result:
        script = cx.load_script(self.cmd.lexeme)
        if isinstance(runnable := script.build(), BuildError):
            return Result.fail(
                short_msg="failed to build generator",
                long_msg=runnable.msg,
            )

        args = self._args_with_seed(cx)
        if isinstance(process := runnable.spawn(args, cwd=cx.cwd), Error):
            return Result.fail(
                short_msg="error when running script",
                long_msg=process.msg,
            )
        # `communicate` reads stdout and stderr together, so a generator that fills one pipe while we
        # wait on the other can't deadlock.
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            msg = ret_code_to_str(process.returncode)
            args_fmt = " ".join(args)
            script_path = utils.relative_to_cwd(cx.script_path(self.cmd.lexeme))
            cmd = f"$ {script_path} {args_fmt}"
            return Result.fail(short_msg=msg, long_msg=f"{cmd}\n{stderr}")

        tests = [test for test in stdout.split(FS) if test]
        if not tests:
            return Result.fail(short_msg="generator didn't produce any output")
        for test in tests:
            cx.next_file(self.group).write_text(test)

        return _success_with_count_result(len(tests))

    def _args_with_seed(self, cx: _CommandCtxt) -> list[str]:
        # We seed the script with the next `idx`, this guarantees it is different
        # every time, even if the script generates more than one file.
        idx = cx.tests_in_group[self.group] + 1
        return [f"{cx.dst_dir.name}-{self.group}-{idx}", *self.args]


def _invalid_command_err(cmd: Token) -> ParseError:
    extensions = typing.get_args(Script.VALID_EXTENSIONS)
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


def validate(subtasks: list[tuple[SubtaskHeader, list[Item]]]) -> list[ValidationError]:
    """Check a parsed testplan for errors that need the whole file, e.g. subtask numbering.

    Subtasks are numbered by their position, so the checks after the first one still make sense
    when a header has the wrong number.
    """
    errors: list[ValidationError] = []
    for i, (header, items) in enumerate(subtasks, start=1):
        if header.number != i:
            errors.append(
                ValidationError(
                    range=header.range,
                    msg=f"found {header}, but [Subtask {i}] was expected",
                ),
            )
        validators = [item for item in items if isinstance(item, Validator)]
        errors.extend(
            ValidationError(
                range=validator.range,
                msg="multiple @validator directives found for the same subtask",
            )
            for validator in validators[1:]
        )

    stns = {Stn(i) for i in range(1, len(subtasks) + 1)}
    graph: dict[Stn, list[Extends]] = {}
    for i, (_, items) in enumerate(subtasks, start=1):
        sti = Stn(i)
        graph[sti] = []
        seen: set[Stn] = set()
        for extends in (item for item in items if isinstance(item, Extends)):
            if extends.stn in seen:
                msg = f"cannot extends twice from the same subtask: `{extends}`"
            elif extends.stn not in stns:
                msg = f"invalid subtask {extends.stn}: `{extends}`"
            elif extends.stn == sti:
                msg = f"a subtask cannot extend itself: `{extends}`"
            else:
                graph[sti].append(extends)
                msg = None
            seen.add(extends.stn)
            if msg is not None:
                errors.append(ValidationError(range=extends.range, msg=msg))
    errors.extend(_cycle_errors(graph))
    return errors


def _cycle_errors(graph: dict[Stn, list[Extends]]) -> list[ValidationError]:
    """Report each `@extends` that closes a cycle in the extends graph."""
    errors: list[ValidationError] = []
    visited: set[Stn] = set()
    stack: set[Stn] = set()

    def dfs(sti: Stn) -> None:
        visited.add(sti)
        stack.add(sti)
        for extends in graph[sti]:
            if extends.stn in stack:
                errors.append(
                    ValidationError(
                        range=extends.range,
                        msg=f"`{extends}` creates a cycle in the extends graph",
                    ),
                )
            elif extends.stn not in visited:
                dfs(extends.stn)
        stack.remove(sti)

    for sti in graph:
        if sti not in visited:
            dfs(sti)
    return errors
