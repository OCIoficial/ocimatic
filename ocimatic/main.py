import argparse
import ctypes
from pathlib import Path
from typing import List, Optional

import ocimatic
from ocimatic import core, server, ui

CONTEST_COMMANDS = ['problemset', 'package']
SINGLE_TASK_COMMANDS = ["run", "build"]
MULTI_TASK_COMMANDS = [
    'check', 'expected', 'pdf', 'compress', 'normalize', 'testplan', 'validate', 'score'
]


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


class CLI:
    contest: core.Contest
    last_dir: Optional[Path]

    def __init__(self) -> None:
        (contest_dir, last_dir) = core.change_directory()
        self.contest = core.Contest(contest_dir)
        self.last_dir = last_dir

    def new_task(self, args: argparse.Namespace) -> None:
        try:
            if Path(self.contest.directory, args.name).exists():
                ui.fatal_error('Cannot create task in existing directory.')
            self.contest.new_task(args.name)
            ui.show_message('Info', f'Task [{args.name}] created')
        except Exception as exc:  # pylint: disable=broad-except
            raise exc
            # ui.fatal_error(r"Couldn't create task: {exc}")

    def run_contest_command(self, args: argparse.Namespace) -> None:
        assert args.command in CONTEST_COMMANDS
        set_verbosity(args, 2)
        if args.command == "problemset":
            self.contest.build_problemset()
        elif args.command == "package":
            self.contest.package()

    def run_single_task_command(self, args: argparse.Namespace) -> None:
        assert args.command in SINGLE_TASK_COMMANDS
        task = self._select_task(args)

        set_verbosity(args, 2)
        if args.command == "run":
            task.run_solution(args.solution)
        elif args.command == "build":
            task.build_solution(args.solution)

    def _select_task(self, args: argparse.Namespace) -> core.Task:
        task = None
        if args.task:
            task = self.contest.find_task(args.task)
        elif self.last_dir:
            task = self.contest.find_task(self.last_dir.name)
        if not task:
            ui.fatal_error(
                "You have to be inside a task or provide the `--task` argument to run this command."
            )
        return task

    def run_multi_task_command(self, args: argparse.Namespace) -> None:
        assert args.command in MULTI_TASK_COMMANDS

        tasks = self._select_tasks(args)

        if not tasks:
            ui.show_message("Warning", "no tasks selected", ui.WARNING)

        for task in tasks:
            if args.command == "check":
                set_verbosity(args, 0)
                task.check_dataset()
            elif args.command == "expected":
                set_verbosity(args, 0 if len(tasks) > 1 else 2)
                task.gen_expected(sample=args.sample, solution=args.solution)
            elif args.command == "pdf":
                set_verbosity(args, 2)
                task.build_statement()
            elif args.command == "compress":
                set_verbosity(args, 2)
                task.compress_dataset(args.random_sort)
            elif args.command == "normalize":
                set_verbosity(args, 2)
                task.normalize()
            elif args.command == "testplan":
                set_verbosity(args, 0 if len(tasks) > 1 else 2)
                task.run_testplan(args.subtask)
            elif args.command == "validate":
                set_verbosity(args, 0 if len(tasks) > 1 else 2)
                task.validate_input(args.subtask)
            elif args.command == "score":
                set_verbosity(args, 2)
                task.score()

    def _select_tasks(self, args: argparse.Namespace) -> List[core.Task]:
        contest = self.contest
        if args.tasks:
            names = args.tasks.split(',')
            tasks: List[core.Task] = []
            for name in names:
                task = contest.find_task(name)
                if task:
                    tasks.append(task)
                else:
                    ui.show_message("Warning", f"cannot find task {name}", ui.WARNING)
            return tasks
        elif self.last_dir:
            task = self.contest.find_task(self.last_dir.name)
            assert task
            return [task]
        else:
            return self.contest.tasks


def set_verbosity(args: argparse.Namespace, value: int) -> None:
    """If `-v` was passed any number of times set that as the verbosity, otherwise use `value`
    as the verbosity"""
    if args.verbosity > 0:
        ocimatic.config['verbosity'] = args.verbosity
    else:
        ocimatic.config['verbosity'] = value


def add_contest_commands(subcommands: argparse._SubParsersAction) -> None:
    # init
    init = subcommands.add_parser("init", help="Initializes a contest in a new directory.")
    init.add_argument("path", help="Path to directory.")
    init.add_argument("--phase")

    # problemset
    subcommands.add_parser("problemset", help="Generate problemset.")

    # package
    subcommands.add_parser("package", help="Generate contest package.")

    # run
    new_task_parser = subcommands.add_parser("new", help="Creates a new task.")
    new_task_parser.add_argument("name", help="Name of the task")


def add_multitask_commands(subcommands: argparse._SubParsersAction) -> None:
    multitask_parser = argparse.ArgumentParser(add_help=False)
    multitask_parser.add_argument('--tasks', help="A comma separated list of tasks.")

    subcommands.add_parser(
        "check",
        help="""Check input/output correcteness by running all correct solutions against all
        test cases and sample inputs""",
        parents=[multitask_parser])

    expected_parser = subcommands.add_parser('expected',
                                             help="""
        Generate expected output by running a correct solution against all the input data.
        By default it will choose any correct solution preferring solutions
        written in C++.
        """,
                                             parents=[multitask_parser])
    expected_parser.add_argument("solution",
                                 nargs="?",
                                 type=Path,
                                 help="""
        A path to a solution. If specified, generate expected output running that solution.""")
    expected_parser.add_argument("--sample",
                                 help="Generate expected output for sample input as well.",
                                 action="store_true",
                                 default=False)

    subcommands.add_parser("pdf", help="Compile the statement's pdf", parents=[multitask_parser])

    compress_parser = subcommands.add_parser("compress",
                                             help="Generate zip file with all test data.",
                                             parents=[multitask_parser])
    compress_parser.add_argument(
        "--random-sort",
        "-r",
        default=False,
        action="store_true",
        help="Add random prefix to output filenames to sort testcases whithin a subtask randomly")

    testplan_parser = subcommands.add_parser("testplan",
                                             help="Run testplan.",
                                             parents=[multitask_parser])
    testplan_parser.add_argument("--subtask", '-st', type=int)

    validate_parser = subcommands.add_parser("validate",
                                             help="Run input validators.",
                                             parents=[multitask_parser])
    validate_parser.add_argument("--subtask", '-st', type=int)

    subcommands.add_parser("score",
                           help="Print the score parameters for cms.",
                           parents=[multitask_parser])

    subcommands.add_parser("normalize",
                           help="Normalize input and output files running dos2unix.",
                           parents=[multitask_parser])


def add_single_task_command(subcommands: argparse._SubParsersAction) -> None:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--task',
        help=
        "The task to run the command in. If not specified it will pick the current task or fail it not inside a task."
    )
    parent_parser.add_argument("solution", help="A path to a solution", type=Path)

    # run
    subcommands.add_parser(
        "run",
        help=
        "Run solutions against all test data and displays the output of the checker and running time.",
        parents=[parent_parser])

    # build
    subcommands.add_parser("build", help="Build a solution.", parents=[parent_parser])


def add_server_command(subcommands: argparse._SubParsersAction) -> None:
    server_parser = subcommands.add_parser(
        "server",
        help="""Start a server which can be used to control ocimatic from a browser.
        This is for the moment very limited, but it's useful during a contest to quickly paste
        and run a solution.
        """)
    server_parser.add_argument("--port", "-p", default="9999", type=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", "-v", action="count", default=0)
    parser.add_argument("--timeout", type=float)

    subcommands = parser.add_subparsers(title="commands", dest="command")

    add_contest_commands(subcommands)
    add_multitask_commands(subcommands)
    add_single_task_command(subcommands)
    add_server_command(subcommands)

    args = parser.parse_args()

    if args.timeout:
        ocimatic.config['timeout'] = args.timeout
    del args.timeout

    if args.command == "init":
        return new_contest(args)
    elif args.command == "server":
        return server.run(args.port)

    cli = CLI()
    if args.command == "new":
        cli.new_task(args)
    elif args.command in CONTEST_COMMANDS:
        cli.run_contest_command(args)
    elif args.command in SINGLE_TASK_COMMANDS:
        cli.run_single_task_command(args)
    elif args.command in MULTI_TASK_COMMANDS:
        cli.run_multi_task_command(args)
    else:
        parser.print_help()
