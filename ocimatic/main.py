import argparse
from pathlib import Path

import ocimatic
from ocimatic import core, server, ui

CONTEST_COMMAND = {'problemset': 'build_problemset', 'package': 'package'}


def new_contest(args: argparse.Namespace) -> None:
    name = args.path

    contest_config: core.ContestConfig = {}
    if args.phase:
        contest_config['phase'] = args.phase

    try:
        contest_path = Path(Path.cwd(), name)
        if contest_path.exists():
            ui.fatal_error("Couldn't create contest. Path already exists")
        core.Contest.create_layout(contest_path, contest_config)
        ui.show_message('Info', 'Contest [%s] created' % name)
    except Exception as exc:  # pylint: disable=broad-except
        ui.fatal_error("Couldn't create contest: %s." % exc)


def contest_mode(args: argparse.Namespace) -> None:
    action = CONTEST_COMMAND[args.command]
    contest_dir = core.change_directory()[0]
    contest = core.Contest(contest_dir)
    getattr(contest, action)()


def new_task(args: argparse.Namespace) -> None:
    try:
        contest_dir = core.change_directory()[0]
        if Path(contest_dir, args.name).exists():
            ui.fatal_error('Cannot create task in existing directory.')
        core.Contest(contest_dir).new_task(args.name)
        ui.show_message('Info', f'Task [{args.name}] created')
    except Exception as exc:  # pylint: disable=broad-except
        ui.fatal_error('Couldn\'t create task: %s' % exc)


TASK_COMMAND = {
    'build': 'build_solutions',
    'check': 'check_dataset',
    'expected': 'gen_expected',
    'pdf': 'build_statement',
    'run': 'run_solutions',
    'compress': 'compress_dataset',
    'normalize': 'normalize',
    'testplan': 'gen_input',
    'validate': 'validate_input',
    'score': 'score'
}


def task_mode(args: argparse.Namespace) -> None:
    (contest_dir, last_dir) = core.change_directory()

    contest = core.Contest(contest_dir)
    if args.command == "check":
        ocimatic.config['verbosity'] -= 1

    method = TASK_COMMAND[args.command]

    if last_dir:
        task = contest.find_task(last_dir.name)
        assert task
        tasks = [task]
    else:
        tasks = contest.tasks
    if not tasks:
        ui.show_message("Warning", "no tasks", ui.WARNING)
    kwargs = vars(args)
    del kwargs['command']
    for task in tasks:
        getattr(task, method)(**vars(args))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", "-v", action="count", default=0)
    parser.add_argument("--timeout", type=float)

    actions = parser.add_subparsers(title="commands", dest="command")

    init = actions.add_parser("init", help="Initializes a contest in a new directory.")
    init.add_argument("path", help="Path to directory.")
    init.add_argument("--phase")
    actions.add_parser("problemset", help="Generate problemset.")
    actions.add_parser("package", help="Generate contest package.")

    new = actions.add_parser("new", help="Creates a new task.")
    new.add_argument("name", help="Name of the task")

    build = actions.add_parser("build",
                               help="""Force a build of all solutions.
        If a pattern is specified, only solutions matching the pattern will be built.""")
    build.add_argument("solution", nargs="?")

    actions.add_parser(
        "check",
        help="""Check input/output correcteness by running all correct solutions against all
        test cases and sample inputs""")

    expected = actions.add_parser('expected',
                                  help="""
        Generate expected output by running a correct solution against all the input data.
        By default it will choose any correct solution preferring solutions
        written in C++.
        """)
    expected.add_argument("pattern",
                          nargs="?",
                          help="""A glob pattern.
        If specified, the first solution matching this pattern will be used to generate the
        expected output.""")
    expected.add_argument("--sample",
                          help="Generate expected output for sample input as well.",
                          action="store_true",
                          default=False)

    actions.add_parser("pdf", help="Compile the statement's pdf")

    run_parser = actions.add_parser(
        "run",
        help=
        "Run solutions against all test data and displays the output of the checker and running time."
    )
    run_parser.add_argument("solution", help="A glob pattern specifying which solution to run")

    compress = actions.add_parser("compress", help="Generate zip file with all test data.")
    compress.add_argument(
        "--random-sort",
        "-r",
        default=False,
        help="Add random prefix to output filenames to sort testcases whithin a subtask randomly")

    testplan_parser = actions.add_parser("testplan", help="Run testplan.")
    testplan_parser.add_argument("--subtask", '-st', type=int)

    validate_parser = actions.add_parser("validate", help="Run input validators.")
    validate_parser.add_argument("--subtask", '-st', type=int)

    actions.add_parser("score", help="Print the score parameters for cms.")

    actions.add_parser("normalize", help="Normalize input and output files running dos2unix.")

    server_parser = actions.add_parser(
        "server", help='Start server which can be used to run solutions in te browser.')
    server_parser.add_argument("--port", "-p", default="9999", type=int)

    args = parser.parse_args()

    if args.timeout:
        ocimatic.config['timeout'] = args.timeout
    del args.timeout

    ocimatic.config['verbosity'] = args.verbosity
    del args.verbosity
    if args.command == "init":
        new_contest(args)
    elif args.command == "new":
        new_task(args)
    elif args.command == "server":
        server.run(args.port)
    elif args.command in CONTEST_COMMAND.keys():
        contest_mode(args)
    elif args.command in TASK_COMMAND.keys():
        task_mode(args)
    else:
        parser.print_help()
