import getopt
import sys
import os

import ocimatic
from ocimatic import core, ui, filesystem
from ocimatic.filesystem import Directory, FilePath

OPTS = {
    'partial': False,
    'task': None,
    'sample': False,
    'timeout': None,
}


def new_contest(args):
    if len(args) < 1:
        ui.fatal_error('You have to specify a name for the contest.')
    name = args[0]

    try:
        cwd = Directory.getcwd()
        if cwd.find(name):
            ui.fatal_error("Couldn't create contest. Path already exists")
        contest_path = FilePath(cwd, name)
        core.Contest.create_layout(contest_path)
        ui.show_message('Info', 'Contest [%s] created' % name)
    except Exception as exc:
        raise
        ui.fatal_error("Couldn't create contest: %s." % exc)


def contest_mode(args):
    if not args:
        ui.ocimatic_help()

    actions = {
        'problemset': 'build_problemset',
        'package': 'package'
    }

    if args[0] == "new":
        new_contest(args[1:])
    elif args[0] in actions:
        contest_dir = filesystem.change_directory()[0]
        contest = core.Contest(contest_dir)
        getattr(contest, actions[args[0]])()
    else:
        ui.fatal_error('Unknown action for contest mode.')


def new_task(args):
    if len(args) < 1:
        ui.fatal_error('You have to specify a name for the task.')
    name = args[0]
    try:
        cwd = Directory.getcwd()
        if cwd.find(name):
            ui.fatal_error('Cannot create task in existing directory.')
        task_dir = cwd.mkdir(name)
        core.Task.create_layout(task_dir)
        ui.show_message('Info', 'Task [%s] created' % name)
    except Exception as exc:
        ui.fatal_error('Couldn\'t create task: %s' % exc)


def task_mode(args):
    if not args:
        ui.ocimatic_help()

    actions = {
        'build': ('build_solutions', ['pattern'], {}),
        'check': ('check_dataset', [], {}),
        'expected': ('gen_expected', ['pattern'], {'sample': OPTS['sample']}),
        'pdf': ('build_statement', [], {}),
        'run': ('run_solutions', ['pattern'], {'partial': OPTS['partial']}),
        'compress': ('compress_dataset', [], {}),
        'normalize': ('normalize', [], {}),
        'gen-input': ('gen_input', [], {}),
        'validate-input': ('validate_input', [], {}),
        'count': ('count', [], {})
    }

    (contest_dir, task_call) = filesystem.change_directory()

    if args[0] == 'new':
        new_task(args[1:])
    elif args[0] in actions:
        contest = core.Contest(contest_dir)
        if OPTS['task']:
            tasks = [contest.find_task(OPTS['task'])]
        elif task_call:
            tasks = [contest.find_task(task_call.basename)]
        else:
            tasks = contest.tasks

        if not tasks:
            ui.show_message("Warning", "no tasks", ui.WARNING)

        # actions[args[0]](tasks, *args[1:])
        action = actions[args[0]]
        args.pop(0)
        kwargs = action[2]
        if len(args) > len(action[1]):
            ui.fatal_error(
                'action %s expects no more than %d argument, %d were given' %
                (action[0], len(action[1]), len(args))
            )
        for (i, arg) in enumerate(args):
            kwargs[action[1][i]] = arg
        for task in tasks:
            getattr(task, action[0])(**kwargs)
    else:
        ui.fatal_error('Unknown action for task mode.')


def dataset_mode(args):
    if not args:
        ui.ocimatic_help()
    actions = {
        'compress': 'compress',
    }
    if args[0] in actions:
        in_ext = '.in'
        sol_ext = '.sol'
        if len(args) > 1:
            in_ext = args[1]
        if len(args) > 2:
            sol_ext = args[2]
        dataset = core.Dataset(Directory.getcwd(), in_ext=in_ext, sol_ext=sol_ext)
        getattr(dataset, actions[args[0]])()
    else:
        ui.fatal_error('Unknown action for dataset mode.')


def main():
    try:
        optlist, args = getopt.gnu_getopt(sys.argv[1:], 'hpst:',
                                          ['help', 'partial', 'task=',
                                           'phase=', 'sample', 'timeout='])
    except getopt.GetoptError as err:
        ui.fatal_error(str(err))

    if len(args) == 0:
        ui.ocimatic_help()

    modes = {
        'contest': contest_mode,
        'task': task_mode,
        'dataset': dataset_mode,
    }

    # If no mode is provided we assume task
    if args[0] in modes:
        mode = args.pop(0)
    else:
        mode = 'task'

    # Process options
    for (key, val) in optlist:
        if key == '-h' or key == '--help':
            ui.ocimatic_help()
        elif key == '--partial' or key == '-p':
            OPTS['partial'] = True
        elif key == '--task' or key == '-t':
            OPTS['task'] = val
        elif key == '--sample' or key == '-s':
            OPTS['sample'] = True
        elif key == '--phase':
            os.environ['OCIMATIC_PHASE'] = val
        elif key == '--timeout':
            ocimatic.config['timeout'] = float(val)

    # Select mode
    # try:
    if mode in modes:
        modes[mode](args)
    else:
        ui.fatal_error('Unknown mode.')
    # except Exception as exc:
    #     ui.fatal_error(str(exc))
