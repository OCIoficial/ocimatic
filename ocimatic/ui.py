import re
import sys
from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import Enum
from typing import Concatenate, Literal, NoReturn, ParamSpec, Protocol, TypeVar, cast

from colorama import Fore, Style

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
    info = "info"

    @staticmethod
    def from_bool(b: bool) -> "Status":  # noqa: FBT001
        return Status.success if b else Status.fail


class IntoWorkResult(Protocol):
    def into_work_result(self) -> "WorkResult":
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkResult:
    status: Status
    short_msg: str
    long_msg: str | None = None

    @staticmethod
    def success(short_msg: str, long_msg: str | None = None) -> "WorkResult":
        return WorkResult(status=Status.success, short_msg=short_msg, long_msg=long_msg)

    @staticmethod
    def fail(short_msg: str, long_msg: str | None = None) -> "WorkResult":
        return WorkResult(status=Status.fail, short_msg=short_msg, long_msg=long_msg)

    @staticmethod
    def info(short_msg: str, long_msg: str | None = None) -> "WorkResult":
        return WorkResult(status=Status.info, short_msg=short_msg, long_msg=long_msg)

    def into_work_result(self) -> "WorkResult":
        return self


@dataclass(frozen=True, kw_only=True, slots=True)
class Result:
    status: Literal[Status.success, Status.fail]
    short_msg: str
    long_msg: str | None = None

    @staticmethod
    def success(short_msg: str, long_msg: str | None = None) -> "Result":
        return Result(status=Status.success, short_msg=short_msg, long_msg=long_msg)

    @staticmethod
    def fail(short_msg: str, long_msg: str | None = None) -> "Result":
        return Result(status=Status.fail, short_msg=short_msg, long_msg=long_msg)

    def into_work_result(self) -> WorkResult:
        return WorkResult(
            status=self.status,
            short_msg=self.short_msg,
            long_msg=self.long_msg,
        )


def write(text: str, color: str = RESET) -> None:
    print(colorize(text, color), end="")


def writeln(text: str = "", color: str = RESET) -> None:
    write(text + "\n", color)


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


def contest_group_header(msg: str) -> None:
    """Print header for a group of works involving a contest."""
    write("\n\n")
    write(colorize(msg, INFO + MAGENTA))
    writeln()


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


def solution_group_footer() -> None:
    writeln()


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


def end_work(result: WorkResult) -> None:
    match result.status:
        case Status.info:
            char = "."
            color = INFO
        case Status.success:
            char = "✓"
            color = OK
        case Status.fail:
            char = "x"
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
