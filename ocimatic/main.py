import getopt
import sys
import os

import ocimatic
from ocimatic import core, ui, filesystem

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
        core.Contest.create_layout(os.path.join(os.getcwd(), name))
        ui.show_message('Info', 'Contest [%s] created' % name)
    except Exception as exc:
        ui.fatal_error('Couldn\'t create contest: %s.' % exc)

def contest_problemset(contest, _):
    contest.build_problemset()

def contest_mode(args):
    if not args:
        ocimatic_help()

    actions = {
        'problemset': contest_problemset,
    }

    if args[0] == "new":
        new_contest(args[1:])
    elif args[0] in actions:
        contest_dir = filesystem.change_directory()[0]
        contest = core.Contest(contest_dir)
        actions[args[0]](contest, args[1:])
    else:
        ui.fatal_error('Unknown action for contest mode.')


def new_task(args):
    if len(args) < 1:
        ui.fatal_error('You have to specify a name for the task.')
    name = args[0]
    try:
        core.Task.create_layout(os.path.join(os.getcwd(), name))
        ui.show_message('Info', 'Task [%s] created' % name)
    except Exception as exc:
        ui.fatal_error('Couldn\'t create task: %s' % exc)

def tasks_run(tasks, _):
    for task in tasks:
        task.run_solutions(partial=OPTS['partial'])


def tasks_check(tasks, _):
    for task in tasks:
        task.check_dataset()


def tasks_build(tasks, _):
    for task in tasks:
        task.build_solutions()


def tasks_gen_expected(tasks, _):
    for task in tasks:
        task.gen_expected(sample=OPTS['sample'])


def tasks_build_statement(tasks, _):
    for task in tasks:
        task.build_statement()

def tasks_compress(tasks, _):
    for task in tasks:
        task.compress_dataset()


def task_mode(args):
    if not args:
        ocimatic_help()

    actions = {
        'build': tasks_build,
        'check': tasks_check,
        'expected': tasks_gen_expected,
        'pdf': tasks_build_statement,
        'run': tasks_run,
        'compress' : tasks_compress,
        # 'normalize' : tasks_normalize,
    }

    (contest_dir, task_call) = filesystem.change_directory()
    contest = core.Contest(contest_dir)

    if args[0] == 'new':
        new_task(args[1:])
    elif args[0] in actions:
        if OPTS['task']:
            tasks = [contest.find_task(OPTS['task'])]
        elif task_call:
            tasks = [contest.find_task(task_call.basename)]
        else:
            tasks = contest.tasks

        if not tasks:
            ui.show_message("Warning", "no tasks", ui.WARNING)

        actions[args[0]](tasks, args[1:])

    else:
        ui.fatal_error('Unknown action for task mode.')

def main():
    try:
        optlist, args = getopt.gnu_getopt(sys.argv[1:], 'hpst:',
                                          ['help', 'partial', 'task=',
                                           'phase=', 'sample', 'timeout='])
    except getopt.GetoptError as err:
        error_message(str(err))

    if len(args) == 0:
        ocimatic_help()

    modes = {
        'contest': contest_mode,
        'task': task_mode,
    }

    # If no mode is provided we assume task
    if args[0] in modes:
        mode = args.pop(0)
    else:
        mode = 'task'

    # Process options
    for (key, val) in optlist:
        if key == '-h' or key == '--help':
            ocimatic_help()
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
            os.environ['timeout'] = val
            OPTS['timeout'] = val

    # Select mode
    # try:
    if mode in modes:
        modes[mode](args)
    else:
        error_message('Unknown mode.')
    # except Exception as exc:
    #     ui.fatal_error(str(exc))
