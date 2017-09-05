import sys
import textwrap
import re
from importlib.util import find_spec

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
    sys.stdout.write(colorize('[%s] %s' % (name, msg), BOLD + YELLOW))
    print()


def workgroup_header(msg, length=35):
    """Header for group of works"""
    print()
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    sys.stdout.write(colorize('[%s]' % (msg), INFO))
    print()


def supergroup_header(msg, length=35):
    """Header for group of works"""
    print()
    print()
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    sys.stdout.write(colorize('[%s]' % (msg), INFO + BLUE))
    print()


def start_work(action, msg, length=45):
    msg = '....' + msg[-length - 4:] if len(msg) - 4 > length else msg
    msg = ' * [' + action + '] ' + msg + '  '
    sys.stdout.write(msg)
    sys.stdout.flush()


def end_work(msg, status):
    color = OK if status else ERROR
    sys.stdout.write(colorize(str(msg), color))
    print()


def fatal_error(message):
    writeln('ocimatic: ' + message)
    writeln()
    # writeln(usage())
    # writeln('Try ' + bold('ocimatic -h') + ' for more information.')
    sys.exit(1)


def show_message(label, msg, color=INFO):
    # sys.stdout.write(' %s \n' % colorize(label + ': ' + msg, color + UNDERLINE))
    sys.stdout.write(' %s \n' % colorize(label + ': ' + msg, color))


def work(action, formatter="{}"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_work(action, formatter.format(*args, **kwargs))
            (st, msg) = func(*args, **kwargs)
            end_work(msg, st)
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
