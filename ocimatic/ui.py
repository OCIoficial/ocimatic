import re
import sys
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import (
    Concatenate,
    Literal,
    NoReturn,
    ParamSpec,
    Protocol,
    TextIO,
    TypeVar,
    cast,
)

from colorama import Fore, Style

_P = ParamSpec("_P")
_S = TypeVar("_S")
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


_VERBOSITY = Verbosity.verbose


def set_verbosity(verbosity: Verbosity) -> None:
    global _VERBOSITY
    _VERBOSITY = verbosity


def colorize(text: str, color: str) -> str:
    return cast(str, color + text + RESET)


def bold(text: str) -> str:
    return colorize(text, BOLD)


def decolorize(text: str) -> str:
    return re.sub(r"\033\[[0-9]+m", "", text)


IO = TextIO
IO_STREAMS: list[IO | None] = [sys.stdout]


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


@contextmanager
def capture_io(stream: IO | None) -> Iterator[None]:
    IO_STREAMS.append(stream)
    yield
    IO_STREAMS.pop()


def write(text: str, color: str = RESET) -> None:
    stream = IO_STREAMS[-1]
    if stream:
        stream.write(colorize(text, color))


def flush() -> None:
    stream = IO_STREAMS[-1]
    if stream:
        stream.flush()


def writeln(text: str = "", color: str = RESET) -> None:
    write(text + "\n", color)


def task_header(name: str, msg: str) -> None:
    """Print header for task."""
    write("\n\n")
    write(colorize(f"[{name}] {msg}", BOLD + Fore.MAGENTA))
    writeln()
    flush()


def workgroup_header(msg: str, length: int = 35) -> None:
    """Print header for a generic group of works."""
    writeln()
    msg = "...." + msg[-length - 4 :] if len(msg) - 4 > length else msg
    color = INFO if _VERBOSITY is Verbosity.verbose else RESET
    write(colorize("[%s]" % (msg), color))
    if _VERBOSITY is Verbosity.verbose:
        writeln()
    else:
        write(" ")
    flush()


def contest_group_header(msg: str, length: int = 35) -> None:
    """Print header for a group of works involving a contest."""
    write("\n\n")
    msg = "...." + msg[-length - 4 :] if len(msg) - 4 > length else msg
    write(colorize("[%s]" % (msg), INFO + YELLOW))
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


def solution_group_header(msg: str, length: int = 40) -> None:
    """Print header for a solution group."""
    writeln()
    msg = "...." + msg[-length - 4 :] if len(msg) - 4 > length else msg
    write(colorize("[%s]" % (msg), INFO + BLUE) + " ")
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
    if _VERBOSITY is Verbosity.quiet:
        return
    msg = "...." + msg[-length - 4 :] if len(msg) - 4 > length else msg
    msg = " * [" + action + "] " + msg + "  "
    write(colorize(msg, CYAN))
    flush()


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
    if _VERBOSITY is Verbosity.verbose:
        write(colorize(str(result.short_msg), color))
        writeln()
    else:
        write(colorize(char, color))
    if result.long_msg and _VERBOSITY is Verbosity.verbose:
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


def workgroup(formatter: str = "{}") -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            workgroup_header(formatter.format(*args, **kwargs))
            return func(*args, **kwargs)

        return wrapper

    return decorator


def task(
    action: str,
) -> Callable[[Callable[Concatenate[_S, _P], _T]], Callable[Concatenate[_S, _P], _T]]:
    def decorator(
        func: Callable[Concatenate[_S, _P], _T],
    ) -> Callable[Concatenate[_S, _P], _T]:
        def wrapper(self: _S, *args: _P.args, **kwargs: _P.kwargs) -> _T:
            task_header(str(self), action)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
