import re
import sys
from contextlib import contextmanager
from typing import (Any, Callable, Iterable, Iterator, List, NamedTuple, NoReturn, Optional, TextIO,
                    TypeVar, cast)

from colorama import Fore, Style

import ocimatic

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


class WorkResult(NamedTuple):
    success: bool
    short_msg: Optional[str] = None
    long_msg: Optional[str] = None


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
    if ocimatic.config['verbosity'] < 0:
        write(' ')
    else:
        writeln()
    flush()


def contest_group_header(msg: str, length: int = 35) -> None:
    """Header for a group of works involving a contest"""
    write('\n\n')
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    write(colorize('[%s]' % (msg), INFO + YELLOW))
    writeln()
    flush()


F1 = TypeVar("F1", bound=Callable[..., Iterable[WorkResult]])


def solution_group(formatter: str = "{}") -> Callable[[F1], Callable[..., None]]:
    def decorator(func: F1) -> Callable[..., None]:
        def wrapper(*args: Any, **kwargs: Any) -> None:
            solution_group_header(formatter.format(*args, **kwargs))
            for result in func(*args, **kwargs):
                end_work(result)
            solution_group_footer()

        return wrapper

    return decorator


def solution_group_header(msg: str, length: int = 35) -> None:
    """Header for a group of works involving a solution"""
    writeln()
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    write(colorize('[%s]' % (msg), INFO + BLUE) + ' ')
    flush()


def solution_group_footer() -> None:
    writeln()
    flush()


F2 = TypeVar('F2', bound=Callable[..., WorkResult])


def work(action: str, formatter: str = "{}") -> Callable[[F2], F2]:
    def decorator(func: F2) -> F2:
        def wrapper(*args: Any, **kwargs: Any) -> WorkResult:
            if not CAPTURE_WORKS:
                start_work(action, formatter.format(*args, **kwargs))
            result = func(*args, **kwargs)
            if CAPTURE_WORKS:
                CAPTURE_WORKS[-1].append(result)
            end_work(result)
            return result

        return cast(F2, wrapper)

    return decorator


def start_work(action: str, msg: str, length: int = 80) -> None:
    if ocimatic.config['verbosity'] < 0:
        return
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    msg = ' * [' + action + '] ' + msg + '  '
    write(colorize(msg, MAGENTA))
    flush()


def end_work(result: WorkResult) -> None:
    color = OK if result.success else ERROR
    if ocimatic.config['verbosity'] < 0:
        write(colorize('.', color))
    else:
        write(colorize(str(result.short_msg), color))
        writeln()
    if result.long_msg and ocimatic.config['verbosity'] > 0:
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


CAPTURE_WORKS: List[List[WorkResult]] = []


@contextmanager
def capture_works() -> Iterator[List[WorkResult]]:
    CAPTURE_WORKS.append([])
    yield CAPTURE_WORKS[-1]
    CAPTURE_WORKS.pop()


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
