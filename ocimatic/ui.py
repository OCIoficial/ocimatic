from __future__ import annotations

import re
import sys
from collections.abc import Callable, Generator
from enum import Enum
from typing import Literal, NoReturn, ParamSpec, TypeVar, cast

from colorama import Fore, Style

from ocimatic.result import IntoWorkResult, Result, Status, WorkResult

_P = ParamSpec("_P")
_T = TypeVar("_T")

RESET: str = Style.RESET_ALL
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
    """In Windows-1252 encoding ✓ is not available, so we use + instead."""
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
    return color + text + RESET


def bold(text: str) -> str:
    return colorize(text, BOLD)


def decolorize(text: str) -> str:
    return re.sub(r"\033\[[0-9]+m", "", text)


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
    # write(" %s \n" % colorize(label + ": " + str(msg), color))
    write(" {} \n".format(colorize(label + ": " + str(msg), color)))


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
