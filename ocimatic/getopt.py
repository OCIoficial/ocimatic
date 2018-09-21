import getopt
from getopt import GetoptError

from ocimatic import ui


def kwargs_from_optlist(action, args, optlist):
    kwargs = _kwargs_from_args(action, args)
    for opt_key, opt_config in action.get('optlist', {}).items():
        long_opt, short_opt = parse_opt_key(opt_key)
        arg_name = opt_config.get('arg_name', long_opt)
        opt_value = optlist.get('--{}'.format(long_opt))
        if not opt_value and short_opt:
            opt_value = optlist.get('-{}'.format(short_opt))
        typ = opt_config.get('type', 'str')
        if opt_value is not None:
            if typ == 'bool':
                kwargs[arg_name] = True
            elif typ == 'num':
                kwargs[arg_name] = int(opt_value)
            else:
                kwargs[arg_name] = opt_value
    return kwargs


def _kwargs_from_args(action, args):
    optional_count = sum(is_optional(arg_name) for arg_name in action.get('args', []))
    _check_args_len(args, action.get('args', []), optional_count)

    kwargs = {}
    curr_arg = 0
    curr_formal_arg = 0
    formal_args = action.get('args', [])
    while curr_arg < len(args):
        formal_arg = formal_args[curr_formal_arg]
        curr_formal_arg += 1

        if is_optional(formal_arg):
            if optional_count <= 0:
                continue
            optional_count -= 1

        arg_name = get_arg_name(formal_arg)
        kwargs[arg_name] = args[curr_arg]
        curr_arg += 1
    return kwargs


def gnu_getopt(args, short_opts, long_opts, *modes):
    long_opts = long_opts.copy()
    for mode in modes:
        mode_short_opts, mode_long_opts = _get_opts(mode)
        short_opts += mode_short_opts
        long_opts.extend(mode_long_opts)
    optlist, args = getopt.gnu_getopt(args, short_opts, long_opts)
    return dict(optlist), args


def is_optional(arg_name):
    return arg_name[-1] == '?'


def _check_args_len(args, action_args, optional_count):
    max_count = len(action_args)
    mandatory_count = max_count - optional_count
    if len(args) > max_count:
        ui.fatal_error(
            'action expects no more than %d argument, %d were given' % (max_count, len(args)))
    if len(args) < mandatory_count:
        ui.fatal_error(
            'action expects at least %d argument, %d were given' % (mandatory_count, len(args)))


def get_arg_name(arg_name):
    if arg_name[-1] == '?':
        return arg_name[:-1]
    return arg_name


def _get_opts(mode):
    short_opts = []
    long_opts = []
    for action in mode.values():
        for opt_key, opt_config in action.get('optlist', {}).items():
            long_opt, short_opt = parse_opt_key(opt_key)
            typ = opt_config.get('type', 'str')
            if typ != 'bool':
                long_opt += '='
            long_opts.append('{}='.format(long_opt) if typ != 'bool' else long_opt)
            if short_opt:
                short_opts.append('{}:'.format(short_opt) if typ != 'bool' else short_opt)
    return ''.join(short_opts), long_opts


def parse_opt_key(opt_key):
    if isinstance(opt_key, tuple):
        return opt_key
    else:
        return opt_key, None
