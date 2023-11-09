from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Concatenate, NoReturn, ParamSpec, Protocol, TypeVar, cast

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
    success = "sucess"
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


def task_header(name: str, msg: str) -> None:
    """Print header for task."""
    write("\n\n")
    write(colorize(f"[{name}] {msg}", BOLD + Fore.MAGENTA))
    writeln()


def workgroup_header(label: str, msg: str | None) -> None:
    """Print header for a generic group of works."""
    writeln()
    color = INFO if _verbosity is Verbosity.verbose else RESET
    write(colorize(f"[{label}]", color))
    if _verbosity is Verbosity.verbose:
        if msg:
            write(colorize(f" {msg}", color))
        writeln()
    else:
        write(" ")
    flush()


def contest_group_header(msg: str) -> None:
    """Print header for a group of works involving a contest."""
    write("\n\n")
    write(colorize(msg, INFO + MAGENTA))
    writeln()
    flush()


SolutionGroup = Generator[Result, None, _T]


def solution_group(
    formatter: str = "{}",
) -> Callable[[Callable[_P, SolutionGroup[_T]]], Callable[_P, _T]]:
    def decorator(func: Callable[_P, SolutionGroup[_T]]) -> Callable[_P, _T]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            solution_group_header(formatter.format(*args, **kwargs))
            gen = func(*args, **kwargs)
            try:
                while True:
                    result = next(gen).into_work_result()
                    end_work(result)
            except StopIteration as exc:
                solution_group_footer()
                return cast(_T, exc.value)

        return wrapper

    return decorator


def solution_group_header(msg: str) -> None:
    """Print header for a solution group."""
    writeln()
    write(colorize(f"[{msg}]", INFO + BLUE) + " ")
    flush()


def solution_group_footer() -> None:
    writeln()
    flush()


_TIntoWorkResult = TypeVar("_TIntoWorkResult", bound=IntoWorkResult)


def work(
    action: str,
    formatter: str = "{}",
) -> Callable[[Callable[_P, _TIntoWorkResult]], Callable[_P, _TIntoWorkResult]]:
    def decorator(
        func: Callable[_P, _TIntoWorkResult],
    ) -> Callable[_P, _TIntoWorkResult]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _TIntoWorkResult:
            start_work(action, formatter.format(*args, **kwargs))
            result = func(*args, **kwargs)
            end_work(result.into_work_result())
            return result

        return wrapper

    return decorator


def start_work(action: str, msg: str, length: int = 80) -> None:
    if _verbosity is Verbosity.quiet:
        return
    msg = "...." + msg[-length - 4 :] if len(msg) - 4 > length else msg
    msg = " * [" + action + "] " + msg + "  "
    write(colorize(msg, CYAN))
    flush()


def end_work(result: WorkResult) -> None:
    match result.status:
        case None:
            char = _INFO_CHAR
            color = INFO
        case Status.success:
            char = _SUCCESS_CHAR
            color = OK
        case Status.fail:
            char = _FAIL_CHAR
            color = ERROR
    if _verbosity is Verbosity.verbose:
        write(colorize(str(result.short_msg), color))
        writeln()
    else:
        write(colorize(char, color))
    if result.long_msg and _verbosity is Verbosity.verbose:
        long_msg = result.long_msg.strip()
        long_msg = "\n".join(f">>> {line}" for line in long_msg.split("\n"))
        write(long_msg)
        writeln()
        writeln()
    flush()


def fatal_error(message: str) -> NoReturn:
    writeln(colorize("ocimatic: " + message, INFO + RED))
    writeln()
    sys.exit(1)


def show_message(label: str, msg: str, color: str = INFO) -> None:
    write(" %s \n" % colorize(label + ": " + str(msg), color))


def contest_group(
    formatter: str = "{}",
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            contest_group_header(formatter.format(*args, **kwargs))
            return func(*args, **kwargs)

        return wrapper

    return decorator


def workgroup(
    label_formatter: str,
    msg: str | None = None,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            workgroup_header(label_formatter.format(*args, **kwargs), msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class HasName(Protocol):
    @property
    def name(self) -> str:
        ...


_THasName = TypeVar("_THasName", bound=HasName)


def task(
    action: str,
) -> Callable[
    [Callable[Concatenate[_THasName, _P], _T]],
    Callable[Concatenate[_THasName, _P], _T],
]:
    def decorator(
        func: Callable[Concatenate[_THasName, _P], _T],
    ) -> Callable[Concatenate[_THasName, _P], _T]:
        def wrapper(self: _THasName, *args: _P.args, **kwargs: _P.kwargs) -> _T:
            task_header(self.name, action)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def relative_to_cwd(path: Path) -> Path:
    commonpath = Path(f"{os.path.commonpath([path, Path.cwd()])}/")
    if commonpath.is_relative_to(ocimatic.config["contest_root"]):
        relpath = os.path.relpath(path, Path.cwd())
        if not relpath.startswith("."):
            relpath = "./" + relpath
        return Path(relpath)
    else:
        return path
