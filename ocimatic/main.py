import getopt
import sys

import ocimatic
from ocimatic import core, filesystem, parseopt, server, ui
from ocimatic.filesystem import Directory, FilePath


def new_contest(args, optlist):
    if not args:
        ui.fatal_error('You have to specify a name for the contest.')
    name = args[0]

    contest_config = {}
    if '--phase' in optlist:
        contest_config['phase'] = optlist['--phase']

    try:
        cwd = Directory.getcwd()
        if cwd.find(name):
            ui.fatal_error("Couldn't create contest. Path already exists")
        contest_path = FilePath(cwd, name)
        core.Contest.create_layout(contest_path, contest_config)
        ui.show_message('Info', 'Contest [%s] created' % name)
    except Exception as exc:  # pylint: disable=broad-except
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
        new_contest(args[1:], optlist)
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
    if not args:
        ui.fatal_error('You have to specify a name for the task.')
    name = args[0]
    try:
        contest_dir = filesystem.change_directory()[0]
        if contest_dir.find(name):
            ui.fatal_error('Cannot create task in existing directory.')
        core.Contest(contest_dir).new_task(name)
        ui.show_message('Info', 'Task [%s] created' % name)
    except Exception as exc:  # pylint: disable=broad-except
        ui.fatal_error('Couldn\'t create task: %s' % exc)


TASK_ACTIONS = {
    'build': {
        'description': 'Force a build of all solutions. If pattern is present only '
        'solutions matching the pattern will be built.',
        'method': 'build_solutions',
        'args': ['pattern?']
    },
    'check': {
        'method':
        'check_dataset',
        'description':
        'Check input/output running all correct solutions with all '
        'testdata and sample inputs.'
    },
    'expected': {
        'description':
        'Generate expected output running some correct solution against '
        'input data. If pattern is provided the expected output is '
        'generated using the first solution matching the pattern.',
        'method':
        'gen_expected',
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
        'description':
        'Run all solutions with all test data and displays the output of '
        'the checker. If pattern is provided only solutions matching the '
        'pattern are considered.',
        'method':
        'run_solutions',
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
        'method': 'compress_dataset',
        'optlist': {
            ('random_sort', 'r'): {
                'type':
                'bool',
                'description':
                'Add random prefix to output filenames to sort testcases whithin a subtask randomly'
            }
        }
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
    'score': {
        'description': 'Print the score parameters for cms.'
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
        kwargs = parseopt.kwargs_from_optlist(action, args, optlist)
        for task in tasks:
            getattr(task, action.get('method', action_name))(**kwargs)
    else:
        ui.fatal_error('Unknown action for task mode.')
        ui.ocimatic_help(TASK_ACTIONS)


DATASET_ACTIONS = {'compress': {'args': ['in_ext?', 'sol_ext?']}}


def dataset_mode(args, optlist):
    if not args:
        ui.ocimatic_help(DATASET_ACTIONS)
    if args[0] in DATASET_ACTIONS:
        action_name = args[0]
        args.pop()
        action = DATASET_ACTIONS[action_name]
        kwargs = parseopt.kwargs_from_optlist(action, args, optlist)
        dataset = core.Dataset(Directory.getcwd())
        getattr(dataset, action.get('method', action_name))(**kwargs)
    else:
        ui.fatal_error('Unknown action for dataset mode.')


SERVER_ACTIONS = {
    'start': {
        'description': 'Start server to run solutions for current contest.',
        'method': 'run',
        'optlist': {
            ('port', 'p'): {
                'type': 'num',
                'description': 'Port where the app will run. default: 9999'
            }
        }
    }
}


def server_mode(args, optlist):
    if not args:
        ui.ocimatic_help(SERVER_ACTIONS)
    if args[0] in SERVER_ACTIONS:
        action_name = args[0]
        args.pop()
        action = SERVER_ACTIONS[action_name]
        kwargs = parseopt.kwargs_from_optlist(action, args, optlist)
        getattr(server, action.get('method', action_name))(**kwargs)
    else:
        ui.fatal_error('Unknown action for server mode.')


def main():
    try:
        optlist, args = parseopt.gnu_getopt(sys.argv[1:], 'hvt:',
                                            ['help', 'task=', 'phase=', 'timeout='], TASK_ACTIONS,
                                            CONTEST_ACTIONS, DATASET_ACTIONS)
    except getopt.GetoptError as err:
        ui.fatal_error(str(err))

    if not args:
        ui.ocimatic_help(TASK_ACTIONS)

    modes = {
        'contest': (contest_mode, CONTEST_ACTIONS),
        'task': (task_mode, TASK_ACTIONS),
        'dataset': (dataset_mode, DATASET_ACTIONS),
        'server': (server_mode, SERVER_ACTIONS),
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
        if key in ('--help', '-h'):
            ui.ocimatic_help(modes[mode][1])
        elif key in ('--task', 't'):
            ocimatic.config['task'] = val
        elif key == '--timeout':
            ocimatic.config['timeout'] = float(val)

    # Select mode
    # try:
    if mode in modes:
        modes[mode][0](args, optlist)
        print()
    else:
        ui.fatal_error('Unknown mode.')
    # except Exception as exc:
    #     ui.fatal_error(str(exc))
