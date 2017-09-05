import sys
import textwrap
import re
from importlib.util import find_spec

import ocimatic
from ocimatic import getopt

if find_spec('colorama') is not None:
    from colorama import Style, Fore
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT
    # UNDERLINE = '\x1b[4m'
    RED = Fore.RED
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
else:
    RESET = ''
    BOLD = ''
    # UNDERLINE = ''
    RED = ''
    GREEN = ''
    YELLOW = ''
    BLUE = ''

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


def writeln(text=''):
    sys.stdout.write(text + '\n')


def write(text):
    sys.stdout.write(text)


def task_header(name, msg):
    """Print header for task"""
    print()
    print()
    write(colorize('[%s] %s' % (name, msg), BOLD + YELLOW))
    print()


def workgroup_header(msg, length=35):
    """Header for group of works"""
    writeln()
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    write(colorize('[%s]' % (msg), INFO))
    writeln()
    sys.stdout.flush()


def workgroup_footer():
    if ocimatic.config['verbosity'] == 0:
        writeln()
        sys.stdout.flush()


def supergroup_header(msg, length=35):
    """Header for group of works"""
    write('\n\n')
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    write(colorize('[%s]' % (msg), INFO + BLUE))
    writeln()
    sys.stdout.flush()


def start_work(action, msg, length=45, verbosity=True):
    if ocimatic.config['verbosity'] == 0 and verbosity:
        return
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    msg = ' * [' + action + '] ' + msg + '  '
    write(msg)
    sys.stdout.flush()


def end_work(msg, status, verbosity=True):
    color = OK if status else ERROR
    if not verbosity or ocimatic.config['verbosity'] > 0:
        write(colorize(str(msg), color))
        writeln()
    else:
        write(colorize('.', color))
    sys.stdout.flush()


def fatal_error(message):
    writeln('ocimatic: ' + message)
    writeln()
    sys.exit(1)


def show_message(label, msg, color=INFO):
    write(' %s \n' % colorize(label + ': ' + msg, color))


def work(action, formatter="{}", verbosity=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_work(action, formatter.format(*args, **kwargs), verbosity=verbosity)
            (st, msg) = func(*args, **kwargs)
            end_work(msg, st, verbosity=verbosity)
            return (st, msg)
        return wrapper
    return decorator


def supergroup(formatter="{}"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            supergroup_header(formatter.format(*args, **kwargs))
            return func(*args, **kwargs)
        return wrapper
    return decorator


def workgroup(formatter="{}"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            workgroup_header(formatter.format(*args, **kwargs))
            res = func(*args, **kwargs)
            workgroup_footer()
            return res
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
                action.get('description', ''), 90,
                subsequent_indent=' ' * 4,
                initial_indent=' ' * 4
            )
        )
        if description.strip():
            writeln(description)
        for opt_key, opt_config in action.get('optlist', {}).items():
            opt = f"{' '*6}{_format_opt(opt_key, opt_config)}   "
            write(opt)
            description = '\n'.join(
                textwrap.wrap(
                    opt_config.get('description', ''),
                    80 - len(opt),
                    subsequent_indent=' ' * len(opt)
                )
            )
            writeln(description)

        writeln()
    sys.exit(0)


def _format_arg(arg):
    arg_name = getopt.get_arg_name(arg)
    if getopt.is_optional(arg):
        return f'[{arg_name}]'
    return arg_name


def _format_opt(opt_key, opt_config):
    long_opt, short_opt = getopt.parse_opt_key(opt_key)
    typ = opt_config.get('type', 'str')
    if typ == 'bool':
        opt = f'--{long_opt}, -{short_opt}' if short_opt else f'--{long_opt}'
    else:
        opt = f'--{long_opt}, -{short_opt}={long_opt}' if short_opt else f'--{long_opt}=long_opt'
    return f'[{opt}]'
