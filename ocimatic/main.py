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
        raise exc
        # ui.fatal_error(r"Couldn't create task: {exc}")


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

    if args.task:
        task = contest.find_task(args.task)
        tasks = [task] if task else []
    elif last_dir:
        task = contest.find_task(last_dir.name)
        assert task
        tasks = [task]
    else:
        tasks = contest.tasks
    if not tasks:
        ui.show_message("Warning", "no tasks", ui.WARNING)
    kwargs = vars(args)
    del kwargs['task']
    del kwargs['command']
    for task in tasks:
        getattr(task, method)(**vars(args))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", "-v", action="count", default=0)
    parser.add_argument("--timeout", type=float)

    subcommands = parser.add_subparsers(title="commands", dest="command")

    # Contest Commands

    init = subcommands.add_parser("init", help="Initializes a contest in a new directory.")
    init.add_argument("path", help="Path to directory.")
    init.add_argument("--phase")

    subcommands.add_parser("problemset", help="Generate problemset.")

    subcommands.add_parser("package", help="Generate contest package.")

    # Task Commands

    task_parent = argparse.ArgumentParser(add_help=False)
    task_parent.add_argument('--task')

    new_task_parser = subcommands.add_parser("new", help="Creates a new task.")
    new_task_parser.add_argument("name", help="Name of the task")

    build_parser = subcommands.add_parser("build",
                                          help="""Force a build of all solutions.
        If a pattern is specified, only solutions matching the pattern will be built.""",
                                          parents=[task_parent])
    build_parser.add_argument("solution", nargs="?", type=Path)

    subcommands.add_parser(
        "check",
        help="""Check input/output correcteness by running all correct solutions against all
        test cases and sample inputs""",
        parents=[task_parent])

    expected_parser = subcommands.add_parser('expected',
                                             help="""
        Generate expected output by running a correct solution against all the input data.
        By default it will choose any correct solution preferring solutions
        written in C++.
        """,
                                             parents=[task_parent])
    expected_parser.add_argument("solution",
                                 nargs="?",
                                 help="""A glob pattern.
        If specified, generate output running this solution.""")
    expected_parser.add_argument("--sample",
                                 help="Generate expected output for sample input as well.",
                                 action="store_true",
                                 default=False)

    subcommands.add_parser("pdf", help="Compile the statement's pdf", parents=[task_parent])

    run_parser = subcommands.add_parser(
        "run",
        help=
        "Run solutions against all test data and displays the output of the checker and running time.",
        parents=[task_parent])
    run_parser.add_argument("solution", help="A path to a solution", type=Path)

    compress_parser = subcommands.add_parser("compress",
                                             help="Generate zip file with all test data.",
                                             parents=[task_parent])
    compress_parser.add_argument(
        "--random-sort",
        "-r",
        default=False,
        help="Add random prefix to output filenames to sort testcases whithin a subtask randomly")

    testplan_parser = subcommands.add_parser("testplan",
                                             help="Run testplan.",
                                             parents=[task_parent])
    testplan_parser.add_argument("--subtask", '-st', type=int)

    validate_parser = subcommands.add_parser("validate",
                                             help="Run input validators.",
                                             parents=[task_parent])
    validate_parser.add_argument("--subtask", '-st', type=int)

    subcommands.add_parser("score",
                           help="Print the score parameters for cms.",
                           parents=[task_parent])

    subcommands.add_parser("normalize",
                           help="Normalize input and output files running dos2unix.",
                           parents=[task_parent])

    # Server

    server_parser = subcommands.add_parser(
        "server",
        help="""Start a server which can be used to control ocimatic from a browser.
        This is for the moment very limited, but it's useful during a contest to quickly paste
        and run a solution.
        """)
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
