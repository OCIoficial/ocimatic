from __future__ import annotations

import argparse
from argparse import ArgumentParser, _SubParsersAction
from collections.abc import Callable
from pathlib import Path

import ocimatic
from ocimatic import core, server, ui

CONTEST_COMMANDS = ["problemset", "package"]
SINGLE_TASK_COMMANDS = ["run", "build"]
MULTI_TASK_COMMANDS = [
    "check-dataset",
    "gen-expected",
    "pdf",
    "compress",
    "normalize",
    "run-testplan",
    "validate-input",
    "score",
]


def new_contest(args: argparse.Namespace) -> None:
    name = args.path

    contest_config: core.ContestConfig = {}
    if args.phase:
        contest_config["phase"] = args.phase

    try:
        contest_path = Path(Path.cwd(), name)
        if contest_path.exists():
            ui.fatal_error("Couldn't create contest. Path already exists")
        core.Contest.create_layout(contest_path, contest_config)
        ui.show_message("Info", "Contest [%s] created" % name)
    except Exception as exc:  # pylint: disable=broad-except
        ui.fatal_error("Couldn't create contest: %s." % exc)


class CLI:
    contest: core.Contest
    last_dir: Path | None

    def __init__(self) -> None:
        (contest_dir, last_dir) = core.find_contest_root()
        self.contest = core.Contest(contest_dir)
        self.last_dir = last_dir

    def new_task(self, args: argparse.Namespace) -> None:
        try:
            if Path(self.contest.directory, args.name).exists():
                ui.fatal_error("Cannot create task in existing directory.")
            self.contest.new_task(args.name)
            ui.show_message("Info", f"Task [{args.name}] created")
        except Exception as exc:  # pylint: disable=broad-except
            raise exc
            # ui.fatal_error(r"Couldn't create task: {exc}")

    def run_contest_command(self, args: argparse.Namespace) -> None:
        assert args.command in CONTEST_COMMANDS
        action: Callable[[core.Contest], None]
        set_verbosity(args, 2)
        if args.command == "problemset":
            action = lambda contest: contest.build_problemset()
        elif args.command == "package":
            action = lambda contest: contest.package()
        else:
            ui.fatal_error(f"invalid contest-task command: `{args.command}`")
        action(self.contest)

    def run_single_task_command(self, args: argparse.Namespace) -> None:
        task = self._select_task(args)

        set_verbosity(args, 2)
        action: Callable[[core.Task], None]
        if args.command == "run":
            action = lambda task: task.run_solution(args.solution, args.timeout)
        elif args.command == "build":
            action = lambda task: task.build_solution(args.solution)
        else:
            ui.fatal_error(f"invalid multi-task command: `{args.command}`")
        action(task)

    def _select_task(self, args: argparse.Namespace) -> core.Task:
        task = None
        if args.task:
            task = self.contest.find_task(args.task)
        elif self.last_dir:
            task = self.contest.find_task(self.last_dir.name)
        if not task:
            ui.fatal_error(
                "You have to be inside a task or provide the `--task` argument to run this command.",
            )
        return task

    def run_multi_task_command(self, args: argparse.Namespace) -> None:
        tasks = self._select_tasks(args)

        if not tasks:
            ui.show_message("Warning", "no tasks selected", ui.WARNING)

        action: Callable[[core.Task], None]
        if args.command == "check-dataset":
            set_verbosity(args, 0)
            failed: list[core.Task] = []
            for task in tasks:
                success = task.check_dataset()
                if not success:
                    failed.append(task)
            if len(tasks) > 1:
                ui.writeln()
                if failed:
                    ui.writeln(
                        "------------------------------------------------",
                        ui.ERROR,
                    )
                    ui.writeln(
                        "Some tasks have issues that need to be resolved.",
                        ui.ERROR,
                    )
                    ui.writeln()
                    ui.writeln("Tasks with issues:", ui.ERROR)
                    for task in failed:
                        ui.writeln(f" * {task.name}", ui.ERROR)
                    ui.writeln(
                        "------------------------------------------------",
                        ui.ERROR,
                    )
                else:
                    ui.writeln("--------------------", ui.OK)
                    ui.writeln("| No issues found! |", ui.OK)
                    ui.writeln("--------------------", ui.OK)
            return
        elif args.command == "gen-expected":
            set_verbosity(args, 0 if len(tasks) > 1 else 2)
            if args.solution and len(tasks) > 1:
                ui.fatal_error(
                    "A solution can only be specified when there's a single target task",
                )
            action = lambda task: task.gen_expected(
                sample=args.sample,
                solution=args.solution,
            )
        elif args.command == "pdf":
            set_verbosity(args, 2)
            action = lambda task: task.build_statement()
        elif args.command == "compress":
            set_verbosity(args, 2)
            action = lambda task: task.compress_dataset(random_sort=args.random_sort)
        elif args.command == "normalize":
            set_verbosity(args, 2)
            action = lambda task: task.normalize()
        elif args.command == "run-testplan":
            set_verbosity(args, 0 if len(tasks) > 1 else 2)
            action = lambda task: task.run_testplan(args.subtask)
        elif args.command == "validate-input":
            set_verbosity(args, 0 if len(tasks) > 1 else 2)
            action = lambda task: task.validate_input(args.subtask)
        elif args.command == "score":
            set_verbosity(args, 2)
            action = lambda task: task.score()
        else:
            ui.fatal_error(f"invalid multi-task command: `{args.command}`")

        for task in tasks:
            action(task)

    def _select_tasks(self, args: argparse.Namespace) -> list[core.Task]:
        contest = self.contest
        if args.tasks:
            names = args.tasks.split(",")
            tasks: list[core.Task] = []
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
    """If `-v` was passed any number of times set that as the verbosity, otherwise use `value` as the verbosity."""
    if args.verbosity > 0:
        ocimatic.config["verbosity"] = args.verbosity
    else:
        ocimatic.config["verbosity"] = value


def add_contest_commands(subcommands: _SubParsersAction[ArgumentParser]) -> None:
    # init
    init = subcommands.add_parser(
        "init",
        help="Initializes a contest in a new directory.",
    )
    init.add_argument("path", help="Path to directory.")
    init.add_argument("--phase")

    # problemset
    subcommands.add_parser("problemset", help="Generate problemset.")

    # package
    subcommands.add_parser("package", help="Generate contest package.")

    # run
    new_task_parser = subcommands.add_parser("new", help="Creates a new task.")
    new_task_parser.add_argument("name", help="Name of the task")


def add_multitask_commands(subcommands: _SubParsersAction[ArgumentParser]) -> None:
    multitask_parser = argparse.ArgumentParser(add_help=False)
    multitask_parser.add_argument("--tasks", help="A comma separated list of tasks.")

    # check-dataset
    subcommands.add_parser(
        "check-dataset",
        help="""Check input/output correcteness by running all correct solutions against all
        test cases and sample inputs. Also check robustness by checking partial solutions pass/fail
        the subtasks they are suppose to.""",
        parents=[multitask_parser],
    )

    # expected
    gen_expected_parser = subcommands.add_parser(
        "gen-expected",
        help="""
        Generate expected output by running a correct solution against all the input data.
        By default it will choose any correct solution preferring solutions
        written in C++.
        """,
        parents=[multitask_parser],
    )
    gen_expected_parser.add_argument(
        "solution",
        nargs="?",
        type=Path,
        help="""
        A path to a solution. If specified, generate expected output running that solution.
        This setting can only be provided if there's a single target task.
        """,
    )
    gen_expected_parser.add_argument(
        "--sample",
        help="Generate expected output for sample input as well.",
        action="store_true",
        default=False,
    )

    # pdf
    subcommands.add_parser(
        "pdf",
        help="Compile the statement's pdf",
        parents=[multitask_parser],
    )

    # compress
    compress_parser = subcommands.add_parser(
        "compress",
        help="Generate zip file with all test data.",
        parents=[multitask_parser],
    )
    compress_parser.add_argument(
        "--random-sort",
        "-r",
        default=False,
        action="store_true",
        help="Add random prefix to output filenames to sort testcases whithin a subtask randomly",
    )

    # testplan
    run_testplan_parser = subcommands.add_parser(
        "run-testplan",
        help="Run testplan.",
        parents=[multitask_parser],
    )
    run_testplan_parser.add_argument("--subtask", "-st", type=int)

    # validate-input
    validate_parser = subcommands.add_parser(
        "validate-input",
        help="Run input validators.",
        parents=[multitask_parser],
    )
    validate_parser.add_argument("--subtask", "-st", type=int)

    # score
    subcommands.add_parser(
        "score",
        help="Print the score parameters for cms.",
        parents=[multitask_parser],
    )

    subcommands.add_parser(
        "normalize",
        help="Normalize input and output files running dos2unix.",
        parents=[multitask_parser],
    )


def add_single_task_command(subcommands: _SubParsersAction[ArgumentParser]) -> None:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--task",
        help="The task to run the command in. If not specified it will pick the current task or fail it not inside a task.",
    )
    parent_parser.add_argument("solution", help="A path to a solution", type=Path)

    # run
    run_parser = subcommands.add_parser(
        "run",
        help="Run solutions against all test data and displays the output of the checker and running time.",
        parents=[parent_parser],
    )
    run_parser.add_argument("--timeout", type=float)

    # build
    subcommands.add_parser("build", help="Build a solution.", parents=[parent_parser])


def add_server_command(subcommands: _SubParsersAction[ArgumentParser]) -> None:
    server_parser = subcommands.add_parser(
        "server",
        help="""Start a server which can be used to control ocimatic from a browser.
        This is for the moment very limited, but it's useful during a contest to quickly paste
        and run a solution.
        """,
    )
    server_parser.add_argument("--port", "-p", default="9999", type=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", "-v", action="count", default=0)

    subcommands = parser.add_subparsers(title="commands", dest="command")

    add_contest_commands(subcommands)
    add_multitask_commands(subcommands)
    add_single_task_command(subcommands)
    add_server_command(subcommands)

    args = parser.parse_args()

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
        parser.print_help()
        parser.print_help()
        parser.print_help()
