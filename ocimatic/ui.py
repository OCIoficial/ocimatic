import re
import sys
import textwrap
from contextlib import contextmanager
from typing import Callable, Iterable, NamedTuple, Optional, TypeVar, cast

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


def colorize(text, color):
    u"Add ANSI coloring to `text`."
    return color + text + RESET


def bold(text):
    return colorize(text, BOLD)


# def underline(text):
#     return colorize(text, UNDERLINE)


def decolorize(text):
    u"Strip ANSI coloring from `text`."
    return re.sub(r'\033\[[0-9]+m', '', text)


IO_STREAMS = [sys.stdout]


class WorkResult(NamedTuple):
    success: bool
    short_msg: Optional[str] = None
    long_msg: Optional[str] = None


@contextmanager
def capture_io(stream):
    IO_STREAMS.append(stream)
    yield IO_STREAMS[-1]
    IO_STREAMS.pop()


def write(text):
    if IO_STREAMS[-1]:
        IO_STREAMS[-1].write(text)


def flush():
    if IO_STREAMS[-1]:
        IO_STREAMS[-1].flush()


def writeln(text=''):
    write(text + '\n')


def task_header(name, msg):
    """Print header for task"""
    write('\n\n')
    write(colorize('[%s] %s' % (name, msg), BOLD + YELLOW))
    writeln()
    flush()


def workgroup_header(msg, length=35):
    """Header for a generic group of works"""
    writeln()
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    write(colorize('[%s]' % (msg), INFO))
    if ocimatic.config['verbosity'] < 0:
        write(' ')
    else:
        writeln()
    flush()


def contest_group_header(msg, length=35):
    """Header for a group of works involving a contest"""
    write('\n\n')
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    write(colorize('[%s]' % (msg), INFO + YELLOW))
    writeln()
    flush()


SolutionGroup = TypeVar("SolutionGroup", bound=Callable[..., Iterable[WorkResult]])


def solution_group(formatter: str = "{}") -> Callable[[SolutionGroup], Callable[..., None]]:
    def decorator(func: SolutionGroup) -> Callable[..., None]:
        def wrapper(*args, **kwargs):
            solution_group_header(formatter.format(*args, **kwargs))
            for result in func(*args, **kwargs):
                end_work(result)
            solution_group_footer()

        return wrapper

    return decorator


def solution_group_header(msg, length=35):
    """Header for a group of works involving a solution"""
    writeln()
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    write(colorize('[%s]' % (msg), INFO + BLUE) + ' ')
    flush()


def solution_group_footer():
    writeln()
    flush()


Work = TypeVar('Work', bound=Callable[..., WorkResult])


def work(action: str, formatter: str = "{}") -> Callable[[Work], Work]:
    def decorator(func: Work) -> Work:
        def wrapper(*args, **kwargs):
            if not CAPTURE_WORKS:
                start_work(action, formatter.format(*args, **kwargs))
            result = func(*args, **kwargs)
            if CAPTURE_WORKS:
                CAPTURE_WORKS[-1].append(result)
            end_work(result)
            return result

        return cast(Work, wrapper)

    return decorator


def start_work(action, msg, length=45):
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


def fatal_error(message):
    writeln(colorize('ocimatic: ' + message, INFO + RED))
    writeln()
    sys.exit(1)


def show_message(label, msg, color=INFO):
    write(' %s \n' % colorize(label + ': ' + str(msg), color))


CAPTURE_WORKS = []


@contextmanager
def capture_works():
    CAPTURE_WORKS.append([])
    yield CAPTURE_WORKS[-1]
    CAPTURE_WORKS.pop()


def contest_group(formatter="{}"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            contest_group_header(formatter.format(*args, **kwargs))
            return func(*args, **kwargs)

        return wrapper

    return decorator


def workgroup(formatter="{}"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            workgroup_header(formatter.format(*args, **kwargs))
            return func(*args, **kwargs)

        return wrapper

    return decorator


def task(action):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            task_header(str(self), action)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
