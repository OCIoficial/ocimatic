import sys
import re
# from importlib.util import find_spec
# if find_spec('colorama') is not None:
#     from colorama import Style, Fore
#     RESET = Style.RESET_ALL
#     BOLD = Style.BRIGHT
#     UNDERLINE = '\x1b[4m'
#     RED = Fore.RED
#     GREEN = Fore.GREEN
#     YELLOW = Fore.YELLOW
#     BLUE = Fore.BLUE
# else:
#     RESET = ''
#     BOLD = ''
#     UNDERLINE = ''
#     RED = ''
#     GREEN = ''
#     YELLOW = ''
#     BLUE = ''

RESET = '\x1b[0m'
BOLD = '\x1b[1m'
UNDERLINE = '\x1b[4m'
RED = '\x1b[31m'
GREEN = '\x1b[32m'
YELLOW = '\x1b[33m'
BLUE = '\x1b[34m'


INFO = BOLD
OK = BOLD + GREEN
WARNING = BOLD + YELLOW
ERROR = BOLD + RED


def colorize(text, color):
    u"Add ANSI coloring to `text`."
    return color + text + RESET


def bold(text):
    return colorize(text, BOLD)


def underline(text):
    return colorize(text, UNDERLINE)


def decolorize(text):
    u"Strip ANSI coloring from `text`."
    return re.sub(r'\033\[[0-9]+m', '', text)


def writeln(text=''):
    sys.stdout.write(text+'\n')


def task_header(name, msg):
    """Print header for task"""
    print()
    print()
    sys.stdout.write(colorize('[%s] %s' % (name, msg), BOLD + YELLOW))
    print()


def workgroup_header(msg, length=35):
    """Header for group of works"""
    print()
    msg = '....' + msg[-length-4:] if len(msg)-4 > length else msg
    sys.stdout.write(colorize('[%s]' % (msg), INFO))
    print()


def start_work(action, msg, length=45):
    msg = '....' + msg[-length-4:] if len(msg)-4 > length else msg
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
    sys.stdout.write(' %s \n' % colorize(label + ': ' + msg, color + UNDERLINE))


def work(action, target=None):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            start_work(action, str(target or self))
            (st, msg) = func(self, *args, **kwargs)
            end_work(msg, st)
            return (st, msg)
        return wrapper
    return decorator


def workgroup(msg=None):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            workgroup_header(msg or str(self))
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def task(action):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            task_header(str(self), action)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def ocimatic_help():
    show_message('INFO', 'Sorry, but we don\'t have a help message yet :(')
    sys.exit(0)

