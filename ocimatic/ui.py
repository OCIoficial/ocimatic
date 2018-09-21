import re
import sys
import textwrap
from contextlib import contextmanager

from colorama import Fore, Style

import ocimatic
from ocimatic import parseopt

RESET = Style.RESET_ALL
BOLD = Style.BRIGHT
# UNDERLINE = '\x1b[4m'
RED = Fore.RED
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
BLUE = Fore.BLUE
MAGENTA = Fore.MAGENTA

# RESET = '\x1b[0m'
# BOLD = '\x1b[1m'
# UNDERLINE = '\x1b[4m'
# RED = '\x1b[31m'
# GREEN = '\x1b[32m'
# YELLOW = '\x1b[33m'
# BLUE = '\x1b[34m'

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
    if ocimatic.config['verbosity'] > 0:
        writeln()
    else:
        write(' ')
    flush()


def contest_group_header(msg, length=35):
    """Header for a group of works involving a contest"""
    write('\n\n')
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    write(colorize('[%s]' % (msg), INFO + MAGENTA))
    writeln()
    flush()


def solution_group_header(msg, length=35):
    """Header for a group of works involving a solution"""
    writeln()
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    write(colorize('[%s]' % (msg), INFO + BLUE) + ' ')
    flush()


def solution_group_footer():
    writeln()
    flush()


def start_work(action, msg, length=45, verbosity=True):
    if ocimatic.config['verbosity'] == 0 and verbosity:
        return
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    msg = ' * [' + action + '] ' + msg + '  '
    write(msg)
    flush()


def end_work(msg, status, verbosity=True):
    color = OK if status else ERROR
    if not verbosity or ocimatic.config['verbosity'] > 0:
        write(colorize(str(msg), color))
        writeln()
    else:
        write(colorize('.', color))
    flush()


def fatal_error(message):
    writeln('ocimatic: ' + message)
    writeln()
    sys.exit(1)


def show_message(label, msg, color=INFO):
    write(' %s \n' % colorize(label + ': ' + msg, color))


CAPTURE_WORKS = []


@contextmanager
def capture_works():
    CAPTURE_WORKS.append([])
    yield CAPTURE_WORKS[-1]
    CAPTURE_WORKS.pop()


def work(action, formatter="{}", verbosity=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not CAPTURE_WORKS:
                start_work(action, formatter.format(*args, **kwargs), verbosity=verbosity)
            (st, msg) = func(*args, **kwargs)
            if CAPTURE_WORKS:
                CAPTURE_WORKS[-1].append((st, msg))
            end_work(msg, st, verbosity=verbosity)
            return (st, msg)

        return wrapper

    return decorator


def solution_group(formatter="{}"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            solution_group_header(formatter.format(*args, **kwargs))
            for st, msg in func(*args, **kwargs):
                end_work(msg, st, verbosity=False)
            solution_group_footer()

        return wrapper

    return decorator


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


def ocimatic_help(mode):
    for action_name, action in mode.items():
        write(' ' * 2 + bold(action_name) + ' ')
        writeln(' '.join(_format_arg(arg) for arg in action.get('args', [])))
        description = '\n'.join(
            textwrap.wrap(
                action.get('description', ''),
                90,
                subsequent_indent=' ' * 4,
                initial_indent=' ' * 4))
        if description.strip():
            writeln(description)
        for opt_key, opt_config in action.get('optlist', {}).items():
            opt = "{}{}   ".format(' ' * 6, _format_opt(opt_key, opt_config))
            write(opt)
            description = '\n'.join(
                textwrap.wrap(
                    opt_config.get('description', ''),
                    80 - len(opt),
                    subsequent_indent=' ' * len(opt)))
            writeln(description)

        writeln()
    sys.exit(0)


def _format_arg(arg):
    arg_name = parseopt.get_arg_name(arg)
    if parseopt.is_optional(arg):
        return '[{}]'.format(arg_name)
    return arg_name


def _format_opt(opt_key, opt_config):
    long_opt, short_opt = parseopt.parse_opt_key(opt_key)
    typ = opt_config.get('type', 'str')
    if typ == 'bool':
        opt = '--{}, -{}'.format(long_opt, short_opt) if short_opt else '--{}'.format(long_opt)
    else:
        if short_opt:
            opt = '--{long_opt}={long_opt}, -{short_opt}={long_opt}'.format(
                long_opt=long_opt, short_opt=short_opt)
        else:
            opt = '--{long_opt}={long_opt}'.format(long_opt=long_opt)
    return '[{}]'.format(opt)
