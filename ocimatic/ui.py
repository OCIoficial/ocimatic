import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (Any, Callable, Generator, Iterator, List, NoReturn, Optional, ParamSpec,
                    Protocol, TextIO, TypeVar, cast)

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
INFO = BOLD
OK = BOLD + GREEN

WARNING = BOLD + YELLOW
ERROR = BOLD + RED


def colorize(text: str, color: str) -> str:
    return cast(str, color + text + RESET)


def bold(text: str) -> str:
    return colorize(text, BOLD)


def decolorize(text: str) -> str:
    return re.sub(r'\033\[[0-9]+m', '', text)


IO = TextIO
IO_STREAMS: List[Optional[IO]] = [sys.stdout]


@dataclass(kw_only=True)
class WorkResult:
    success: bool
    short_msg: str
    long_msg: Optional[str] = None

    def into_work_result(self) -> 'WorkResult':
        return self


class IntoWorkResult(Protocol):

    def into_work_result(self) -> WorkResult:
        ...


@contextmanager
def capture_io(stream: Optional[IO]) -> Iterator[None]:
    IO_STREAMS.append(stream)
    yield
    IO_STREAMS.pop()


def write(text: str) -> None:
    stream = IO_STREAMS[-1]
    if stream:
        stream.write(text)


def flush() -> None:
    stream = IO_STREAMS[-1]
    if stream:
        stream.flush()


def writeln(text: str = '') -> None:
    write(text + '\n')


def task_header(name: str, msg: str) -> None:
    """Print header for task"""
    write('\n\n')
    write(colorize('[%s] %s' % (name, msg), BOLD + YELLOW))
    writeln()
    flush()


def workgroup_header(msg: str, length: int = 35) -> None:
    """Header for a generic group of works"""
    writeln()
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    write(colorize('[%s]' % (msg), INFO))
    if ocimatic.config['verbosity'] > 0:
        writeln()
    else:
        write(' ')
    flush()


def contest_group_header(msg: str, length: int = 35) -> None:
    """Header for a group of works involving a contest"""
    write('\n\n')
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    write(colorize('[%s]' % (msg), INFO + YELLOW))
    writeln()
    flush()


SolutionGroup = Generator[WorkResult, None, _T]


def solution_group(
        formatter: str = "{}") -> Callable[[Callable[_P, SolutionGroup[_T]]], Callable[_P, _T]]:

    def decorator(func: Callable[_P, SolutionGroup[_T]]) -> Callable[_P, _T]:

        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            solution_group_header(formatter.format(*args, **kwargs))
            gen = func(*args, **kwargs)
            try:
                while True:
                    result = next(gen)
                    end_work(result)
            except StopIteration as exc:
                solution_group_footer()
                return cast(_T, exc.value)

        return wrapper

    return decorator


def solution_group_header(msg: str, length: int = 40) -> None:
    """Header for a group of works involving a solution"""
    writeln()
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    write(colorize('[%s]' % (msg), INFO + BLUE) + ' ')
    flush()


def solution_group_footer() -> None:
    writeln()
    flush()


_TIntoWorkResult = TypeVar('_TIntoWorkResult', bound=IntoWorkResult)


def work(
    action: str,
    formatter: str = "{}"
) -> Callable[[Callable[_P, _TIntoWorkResult]], Callable[_P, _TIntoWorkResult]]:

    def decorator(func: Callable[_P, _TIntoWorkResult]) -> Callable[_P, _TIntoWorkResult]:

        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _TIntoWorkResult:
            start_work(action, formatter.format(*args, **kwargs))
            result = func(*args, **kwargs)
            end_work(result.into_work_result())
            return result

        return wrapper

    return decorator


def start_work(action: str, msg: str, length: int = 80) -> None:
    if ocimatic.config['verbosity'] == 0:
        return
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    msg = ' * [' + action + '] ' + msg + '  '
    write(colorize(msg, MAGENTA))
    flush()


def end_work(result: WorkResult) -> None:
    color = OK if result.success else ERROR
    if ocimatic.config['verbosity'] > 0:
        write(colorize(str(result.short_msg), color))
        writeln()
    else:
        write(colorize("." if result.success else "⨯", color))
    if result.long_msg and ocimatic.config['verbosity'] > 1:
        long_msg = result.long_msg.strip()
        long_msg = "\n".join(f">>> {line}" for line in long_msg.split("\n"))
        write(long_msg)
        writeln()
        writeln()
    flush()


def fatal_error(message: str) -> NoReturn:
    writeln(colorize('ocimatic: ' + message, INFO + RED))
    writeln()
    sys.exit(1)


def show_message(label: str, msg: str, color: str = INFO) -> None:
    write(' %s \n' % colorize(label + ': ' + str(msg), color))


F = TypeVar("F", bound=Callable[..., Any])


def contest_group(formatter: str = "{}") -> Callable[[F], F]:

    def decorator(func: F) -> F:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            contest_group_header(formatter.format(*args, **kwargs))
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def workgroup(formatter: str = "{}") -> Callable[[F], F]:

    def decorator(func: F) -> F:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            workgroup_header(formatter.format(*args, **kwargs))
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def task(action: str) -> Callable[[F], F]:

    def decorator(func: F) -> F:

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            task_header(str(self), action)
            return func(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator
