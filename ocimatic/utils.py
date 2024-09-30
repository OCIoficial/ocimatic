from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, Literal, NoReturn, ParamSpec, Protocol, TypeVar, cast

from colorama import Fore, Style

import ocimatic

_P = ParamSpec("_P")
_T = TypeVar("_T")

RESET = Style.RESET_ALL
BOLD = Style.BRIGHT
RED = Fore.RED
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
BLUE = Fore.BLUE
MAGENTA = Fore.MAGENTA
CYAN = Fore.CYAN
INFO = BOLD
OK = BOLD + GREEN

WARNING = BOLD + YELLOW
ERROR = BOLD + RED


def _success_char() -> str:
    """In windows Windows-1252 encoding ✓ is not available, so we use + instead."""
    try:
        "✓".encode(sys.stdout.encoding)
        return "✓"
    except UnicodeEncodeError:
        return "+"


_INFO_CHAR = "."
_FAIL_CHAR = "x"
_SUCCESS_CHAR = _success_char()


class Verbosity(Enum):
    quiet = 0
    verbose = 2


_verbosity = Verbosity.verbose


def set_verbosity(verbosity: Verbosity) -> None:
    global _verbosity
    _verbosity = verbosity


def colorize(text: str, color: str) -> str:
    # pyright doesn't like the cast here, but it's needed by mypy
    return cast(str, color + text + RESET)  # pyright: ignore [reportUnnecessaryCast]


def bold(text: str) -> str:
    return colorize(text, BOLD)


def decolorize(text: str) -> str:
    return re.sub(r"\033\[[0-9]+m", "", text)


class Status(Enum):
    success = "success"
    fail = "fail"

    @staticmethod
    def from_bool(b: bool) -> Status:  # noqa: FBT001
        return Status.success if b else Status.fail

    def to_bool(self) -> bool:
        match self:
            case Status.success:
                return True
            case Status.fail:
                return False

    def __iand__(self, other: Status) -> Status:
        return Status.from_bool(self.to_bool() and other.to_bool())


class IntoWorkResult(Protocol):
    def into_work_result(self) -> WorkResult:
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkResult:
    status: Status | None
    short_msg: str
    long_msg: str | None = None

    @staticmethod
    def success(short_msg: str, long_msg: str | None = None) -> WorkResult:
        return WorkResult(status=Status.success, short_msg=short_msg, long_msg=long_msg)

    @staticmethod
    def fail(short_msg: str, long_msg: str | None = None) -> WorkResult:
        return WorkResult(status=Status.fail, short_msg=short_msg, long_msg=long_msg)

    @staticmethod
    def info(short_msg: str, long_msg: str | None = None) -> WorkResult:
        return WorkResult(status=None, short_msg=short_msg, long_msg=long_msg)

    def into_work_result(self) -> WorkResult:
        return self


@dataclass(frozen=True, slots=True)
class Error:
    msg: str


@dataclass(frozen=True, kw_only=True, slots=True)
class Result:
    status: Status
    short_msg: str
    long_msg: str | None = None

    @staticmethod
    def success(short_msg: str, long_msg: str | None = None) -> Result:
        return Result(status=Status.success, short_msg=short_msg, long_msg=long_msg)

    @staticmethod
    def fail(short_msg: str, long_msg: str | None = None) -> Result:
        return Result(status=Status.fail, short_msg=short_msg, long_msg=long_msg)

    def is_fail(self) -> bool:
        return self.status == Status.fail

    def into_work_result(self) -> WorkResult:
        return WorkResult(
            status=self.status,
            short_msg=self.short_msg,
            long_msg=self.long_msg,
        )


def write(text: str, color: str = RESET) -> None:
    sys.stdout.write(colorize(text, color))


def flush() -> None:
    sys.stdout.flush()


def writeln(text: str = "", color: str = RESET) -> None:
    write(text + "\n", color)
    flush()


def fatal_error(message: str) -> NoReturn:
    writeln(colorize("ocimatic: " + message, INFO + RED))
    writeln()
    sys.exit(1)


def show_message(label: str, msg: str, color: str = INFO) -> None:
    write(" %s \n" % colorize(label + ": " + str(msg), color))


_TIntoWorkResult = TypeVar("_TIntoWorkResult", bound=IntoWorkResult)


def work(
    action: str,
    formatter: str = "{}",
) -> Callable[[Callable[_P, _TIntoWorkResult]], Callable[_P, _TIntoWorkResult]]:
    def decorator(
        func: Callable[_P, _TIntoWorkResult],
    ) -> Callable[_P, _TIntoWorkResult]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _TIntoWorkResult:
            _start_work(action, formatter.format(*args, **kwargs))
            result = func(*args, **kwargs)
            _end_work(result.into_work_result(), _verbosity)
            return result

        return wrapper

    return decorator


def _start_work(action: str, msg: str, length: int = 80) -> None:
    if _verbosity is Verbosity.quiet:
        return
    msg = "...." + msg[-length - 4 :] if len(msg) - 4 > length else msg
    msg = " * [" + action + "] " + msg + "  "
    write(colorize(msg, CYAN))
    flush()


def _end_work(result: WorkResult, verbosity: Verbosity) -> None:
    match result.status:
        case None:
            char = _INFO_CHAR
            color = RESET
        case Status.success:
            char = _SUCCESS_CHAR
            color = GREEN
        case Status.fail:
            char = _FAIL_CHAR
            color = RED
    if verbosity is Verbosity.verbose:
        write(colorize(str(result.short_msg), color))
        writeln()

        if result.long_msg:
            long_msg = result.long_msg.strip()
            long_msg = "\n".join(f">>> {line}" for line in long_msg.split("\n"))
            write(long_msg)
            writeln()
    else:
        write(colorize(char, color))
    flush()


def hd(
    level: Literal[1, 2],
    label_formatter: str,
    msg: str | None = None,
    color: str | None = None,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            _fmt_header(level, label_formatter.format(*args, **kwargs), msg, color)
            result = func(*args, **kwargs)
            _fmt_footer(level)
            return result

        return wrapper

    return decorator


def hd1(
    label_formatter: str,
    msg: str | None = None,
    color: str = RESET,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    return hd(1, label_formatter, msg, color)


def hd2(
    label_formatter: str,
    msg: str | None = None,
    color: str | None = None,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    return hd(2, label_formatter, msg, color)


WorkHd = Generator[Result, None, _T]


def workhd(
    formatter: str = "{}",
    color: str | None = None,
) -> Callable[[Callable[_P, WorkHd[_T]]], Callable[_P, _T]]:
    def decorator(func: Callable[_P, WorkHd[_T]]) -> Callable[_P, _T]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            _start_workhd(formatter.format(*args, **kwargs), color=color)
            gen = func(*args, **kwargs)
            try:
                while True:
                    result = next(gen).into_work_result()
                    _end_work(result, Verbosity.verbose)
            except StopIteration as exc:
                _fmt_footer(1)
                return cast(_T, exc.value)

        return wrapper

    return decorator


def _fmt_header(
    level: Literal[1, 2],
    label: str,
    msg: str | None,
    color: str | None = None,
) -> None:
    if level == 1 or _verbosity is Verbosity.verbose:
        writeln()
    write(colorize(f"[{label}]", color or RESET))
    if msg:
        write(colorize(f" {msg}", color or RESET))

    if level == 2 and _verbosity is Verbosity.quiet:
        write(" ")
    else:
        writeln()
    flush()


def _start_workhd(label: str, color: str | None = None) -> None:
    writeln()
    write(colorize(f"[{label}] ", color or RESET))


def _fmt_footer(level: Literal[1, 2]) -> None:
    if level == 2 and _verbosity is Verbosity.quiet:
        writeln()
        flush()


def relative_to_cwd(path: Path) -> str:
    commonpath = Path(os.path.commonpath([path, Path.cwd()]))
    if commonpath.is_relative_to(ocimatic.contest_root):
        relpath = os.path.relpath(path, Path.cwd())
        if not relpath.startswith("."):
            relpath = "." + os.path.sep + relpath
        return relpath
    else:
        return str(path)


class Comparable(Protocol):
    def __lt__(self: _T, __other: _T) -> bool:
        ...


_K = TypeVar("_K", bound=Comparable)
_V = TypeVar("_V")


class SortedDict(Generic[_K, _V]):
    """A dict that iterates over keys in sorted order."""

    def __init__(self, iter: Iterable[tuple[_K, _V]] | None = None) -> None:
        self._dict = dict(iter or [])

    def __getitem__(self, key: _K) -> _V:
        return self._dict[key]

    def __setitem__(self, key: _K, val: _V) -> None:
        self._dict[key] = val

    def __contains__(self, key: _K) -> bool:
        return key in self._dict

    def __repr__(self) -> str:
        items = ", ".join(f"{key!r}: {val!r}" for key, val in self.items())
        return f"{{{items}}}"

    def __len__(self) -> int:
        return len(self._dict)

    def setdefault(self, key: _K, default: _V) -> _V:
        return self._dict.setdefault(key, default)

    def keys(self) -> list[_K]:
        return sorted(self._dict.keys())

    def values(self) -> Iterator[_V]:
        for key in self.keys():
            yield self._dict[key]

    def items(self) -> Iterator[tuple[_K, _V]]:
        for key in self.keys():
            yield (key, self._dict[key])

    def __iter__(self) -> Iterator[_K]:
        yield from self.keys()


class Stn:
    """A wrapper over an integer used as an identifier for a subtask."""

    def __init__(self, stn: int) -> None:
        self._idx = stn

    def __hash__(self) -> int:
        return hash(self._idx)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stn):
            raise ValueError(f"Cannot compare Stn with {type(other)}")
        return self._idx == other._idx

    def __str__(self) -> str:
        return f"{self._idx}"

    def __repr__(self) -> str:
        return f"st{self._idx}"

    def __lt__(self, other: Stn) -> bool:
        return self._idx < other._idx
