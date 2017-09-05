import sys
import os

import ocimatic
from ocimatic import core, ui, filesystem, getopt
from ocimatic.filesystem import Directory, FilePath


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
        ui.fatal_error("Couldn't create contest: %s." % exc)


CONTEST_ACTIONS = {
    'problemset': {
        'description': 'Generate problemset.',
        'method': 'build_problemset'
    },
    'package': {
        'description': 'Generate contest package.',
        'method': 'package'
    }
}


def contest_mode(args, optlist):
    if not args:
        ui.ocimatic_help(CONTEST_ACTIONS)

    if args[0] == "new":
        new_contest(args[1:])
    elif args[0] in CONTEST_ACTIONS:
        action_name = args[0]
        args.pop(0)
        action = CONTEST_ACTIONS[action_name]
        contest_dir = filesystem.change_directory()[0]
        contest = core.Contest(contest_dir)
        getattr(contest, action.get('method', action_name))()
    else:
        ui.fatal_error('Unknown action for contest mode.')
        ui.ocimatic_help(CONTEST_ACTIONS)


def new_task(args):
    if len(args) < 1:
        ui.fatal_error('You have to specify a name for the task.')
    name = args[0]
    try:
        cwd = Directory.getcwd()
        if cwd.find(name):
            ui.fatal_error('Cannot create task in existing directory.')
        task_dir = FilePath(cwd, name)
        core.Task.create_layout(task_dir)
        ui.show_message('Info', 'Task [%s] created' % name)
    except Exception as exc:
        ui.fatal_error('Couldn\'t create task: %s' % exc)


TASK_ACTIONS = {
    'build': {
        'description': 'Force a build of all solutions. If pattern is present only '
                       'solutions matching the pattern will be built.',
        'method': 'build_solutions',
        'args': ['pattern?']
    },
    'check': {
        'method': 'check_dataset',
        'description': 'Check input/output running all correct solutions with all '
                       'testdata and sample inputs.'
    },
    'expected': {
        'description': 'Generate expected output running some correct solution against '
                       'input data. If pattern is provided the expected output is '
                       'generated using the first solution matching the pattern.',
        'method': 'gen_expected',
        'args': ['pattern?'],
        'optlist': {
            ('sample', 's'): {
                'type': 'bool',
                'description': 'Generate expected output for sample input as well.'
            }
        }
    },
    'pdf': {
        'description': 'Compile statement pdf',
        'method': 'build_statement'
    },
    'run': {
        'description': 'Run all solutions with all test data and displays the output of '
                       'the checker. If pattern is provided only solutions matching the '
                       'pattern are considered.',
        'method': 'run_solutions',
        'args': ['pattern?'],
        'optlist': {
            ('partial', 'p'): {
                'type': 'bool',
                'description': 'Consider partial solutions as well.'
            }
        },
    },
    'compress': {
        'description': 'Generate zip file with all test data.',
        'method': 'compress_dataset'
    },
    'normalize': {
        'description': 'Normalize input and output with dos2unix.'
    },
    'input': {
        'description': 'Run testplan.',
        'method': 'gen_input'
    },
    'validate-input': {
        'description': 'Run input checkers.',
        'method': 'validate_input'
    },
    'count': {
        'description': 'Count number of input tests'
    }
}


def task_mode(args, optlist):
    if not args:
        ui.ocimatic_help(TASK_ACTIONS)

    (contest_dir, task_call) = filesystem.change_directory()

    if args[0] == 'new':
        new_task(args[1:])
    elif args[0] in TASK_ACTIONS:
        contest = core.Contest(contest_dir)
        if ocimatic.config['task']:
            tasks = [contest.find_task(ocimatic.config['task'])]
        elif task_call:
            tasks = [contest.find_task(task_call.basename)]
        else:
            tasks = contest.tasks

        if not tasks:
            ui.show_message("Warning", "no tasks", ui.WARNING)

        action_name = args[0]
        args.pop(0)
        action = TASK_ACTIONS[action_name]
        kwargs = getopt.kwargs_from_optlist(action, args, optlist)
        for task in tasks:
            getattr(task, action.get('method', action_name))(**kwargs)
    else:
        ui.fatal_error('Unknown action for task mode.')
        ui.ocimatic_help(TASK_ACTIONS)


DATASET_ACTIONS = {'compress': {'args': ['in_ext?', 'sol_ext?']}}


def dataset_mode(args, optlist):
    if not args:
        ui.ocimatic_help(DATASET_ACTIONS)
    if args[0] in actions:
        action_name = args[0]
        args.pop()
        action = DATASET_ACTIONS[action_name]
        kwargs = getopt.kwargs_from_optlist(action, args, optlist)
        dataset = core.Dataset(Directory.getcwd())
        getattr(dataset, action.get('method', action_name))(**kwargs)
    else:
        ui.fatal_error('Unknown action for dataset mode.')


def main():
    try:
        optlist, args = getopt.gnu_getopt(sys.argv[1:], 'hvt:',
                                          ['help', 'task=', 'phase=', 'timeout='],
                                          TASK_ACTIONS, CONTEST_ACTIONS, DATASET_ACTIONS)
    except getopt.GetoptError as err:
        ui.fatal_error(str(err))

    if len(args) == 0:
        ui.ocimatic_help(TASK_ACTIONS)

    modes = {
        'contest': (contest_mode, CONTEST_ACTIONS),
        'task': (task_mode, TASK_ACTIONS),
        'dataset': (dataset_mode, DATASET_ACTIONS)
    }

    # If no mode is provided we assume task
    if args[0] in modes:
        mode = args.pop(0)
    else:
        mode = 'task'

    # Process options
    for key, val in optlist.items():
        if key == '-v':
            ocimatic.config['verbosity'] += 1
        if key == '-h' or key == '--help':
            ui.ocimatic_help(modes[mode][1])
        elif key == '--task' or key == '-t':
            ocimatic.config['task'] = val
        elif key == '--phase':
            os.environ['OCIMATIC_PHASE'] = val
        elif key == '--timeout':
            ocimatic.config['timeout'] = float(val)

    # Select mode
    # try:
    if mode in modes:
        modes[mode][0](args, optlist)
    else:
        ui.fatal_error('Unknown mode.')
    # except Exception as exc:
    #     ui.fatal_error(str(exc))
