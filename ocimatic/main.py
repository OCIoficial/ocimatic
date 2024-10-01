from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, NoReturn

import click
import cloup
from click.shell_completion import CompletionItem
from cloup.constraints import If, accept_none, mutually_exclusive

import ocimatic
from ocimatic import core, server, utils
from ocimatic.utils import Stn


class CLI:
    def __init__(self) -> None:
        self._data: tuple[core.Contest, Path | None] | None = None

    @property
    def contest(self) -> core.Contest:
        (contest, _) = self._load()
        return contest

    @property
    def last_dir(self) -> core.Path | None:
        (_, last_dir) = self._load()
        return last_dir

    def _load(self) -> tuple[core.Contest, Path | None]:
        if not self._data:
            result = core.find_contest_root()
            if not result:
                utils.fatal_error("ocimatic was not called inside a contest.")
            self._data = (core.Contest(result[0]), result[1])
        return self._data

    def new_task(self, name: str) -> None:
        if Path(self.contest.directory, name).exists():
            utils.fatal_error("Cannot create task in existing directory.")
        self.contest.new_task(name)
        utils.show_message("Info", f"Task [{name}] created", utils.OK)

    def select_task(self, name: str | None) -> core.Task | None:
        task = None
        if name is not None:
            task = self.contest.find_task_by_name(name)
        elif self.last_dir:
            task = self.contest.find_task_by_dir(self.last_dir)
        return task

    def select_tasks(self) -> list[core.Task]:
        task = None
        if self.last_dir:
            task = self.contest.find_task_by_dir(self.last_dir)
        if task is not None:
            return [task]
        else:
            return self.contest.tasks


_SOLUTION_HELP = (
    "If the path is absolute, load solution directly from the path. "
    "If the path is relative, try finding the solution relative to the following locations "
    "until a match is found (or we fail to find one): '<task>/solutions/correct', "
    "'<task>/solutions/partial', '<task>/solutions/', and '<cwd>'. "
    "Here <task> refers to the path of the current task and <cwd> to the current working directory."
)


def _solution_completion(
    *,
    partial: bool = True,
) -> Callable[[click.Context, click.Parameter, str], list[CompletionItem]]:
    def inner(
        ctx: click.Context,
        param: click.Parameter,
        incomplete: str,
    ) -> list[CompletionItem]:
        try:
            del param
            data = core.find_contest_root()
            if not data:
                return []

            task_name: str | None = ctx.params.get("task_name")

            task = None
            if task_name is not None:
                task = core.Contest.load_task_by_name(data[0], task_name)
            elif data[1] is not None:
                task = core.Contest.load_task_by_dir(data[0], data[1])

            if not task:
                return []

            return task.solution_completion(incomplete, partial=partial)
        except Exception:
            return []

    return inner


@cloup.command(help="Initialize a contest in a new directory")
@cloup.argument("path", help="Path to directory")
@cloup.option("--phase")
def init(path: str, phase: str | None) -> None:
    try:
        contest_path = Path(Path.cwd(), path)
        if contest_path.exists():
            utils.fatal_error("Couldn't create contest. Path already exists")
        core.Contest.create_layout(contest_path, phase)
        utils.show_message("Info", f"Contest [{path}] created", utils.OK)
    except Exception as exc:
        utils.fatal_error("Couldn't create contest: %s." % exc)


@cloup.command(
    "server",
    short_help="Start a server to control ocimatic from a browser",
    help="Start a server which can be used to control ocimatic from a browser. "
    "This is for the moment very limited, but it's useful during a contest to quickly paste "
    "and run a solution.",
)
@cloup.option("--port", "-p", default="9999", type=int)
@cloup.pass_obj
def run_server(cli: CLI, port: int) -> None:
    server.run(cli.contest, port)


@cloup.command(help="Generate problemset pdf")
@cloup.pass_obj
def problemset(cli: CLI) -> None:
    status = cli.contest.build_problemset()
    exit_with_status(status)


@cloup.command(
    short_help="Make a zip archive of the contest",
    help="Make an archive of the contest containing the statements and dataset.",
)
@cloup.pass_obj
def archive(cli: CLI) -> None:
    cli.contest.archive()


def _validate_task_name(ctx: click.Context, param: click.Argument, value: str) -> str:
    del ctx, param
    if not value.isalpha():
        raise click.BadParameter("Task name must be a word containing only letters.")
    return value


@cloup.command(help="Create a new task")
@cloup.argument("name", help="Name of the task", callback=_validate_task_name)
@cloup.pass_obj
def new_task(cli: CLI, name: str) -> None:
    cli.new_task(name)


@cloup.command(
    short_help="Run dataset validation",
    help="""
Runs multiple validations on the dataset:\n
 - Check input/output correcteness by running all correct solutions against all test cases.\n
 - Test robustness by running partial solutions and verify they fail the subtasks they are suppose to.\n
 - Validate input/output formatting ensuring basic rules like lines not ending with spaces.\n
 - Run all input validators.\n
""",
)
@cloup.pass_obj
def check_dataset(cli: CLI) -> None:
    utils.set_verbosity(utils.Verbosity.quiet)
    tasks = cli.select_tasks()
    failed = [task for task in tasks if task.check_dataset() == utils.Status.fail]
    if len(tasks) > 1:
        utils.writeln()
        if failed:
            utils.writeln(
                "------------------------------------------------",
                utils.ERROR,
            )
            utils.writeln(
                "Some tasks have issues that need to be resolved.",
                utils.ERROR,
            )
            utils.writeln()
            utils.writeln("Tasks with issues:", utils.ERROR)
            for task in failed:
                utils.writeln(f" * {task.name}", utils.ERROR)
            utils.writeln(
                "------------------------------------------------",
                utils.ERROR,
            )
        else:
            utils.writeln("--------------------", utils.OK)
            utils.writeln("| No issues found! |", utils.OK)
            utils.writeln("--------------------", utils.OK)

    if len(failed) > 0:
        exit_with_status(utils.Status.fail)


@cloup.command(
    short_help="Generate expected output",
    help="Generate expected output by running a correct solution against all the input data. "
    "By default it will choose any correct solution preferring solutions "
    "written in C++.",
)
@cloup.option(
    "--solution",
    required=False,
    help="A path to a solution. If specified, generate expected output running that solution. "
    "This option can only be used when running the command on a single task. "
    + _SOLUTION_HELP,
    type=click.Path(),
    shell_complete=_solution_completion(partial=False),
)
@cloup.option(
    "--sample",
    help="Generate expected output for sample input as well",
    is_flag=True,
    default=False,
)
@cloup.pass_obj
def gen_expected(cli: CLI, solution: str | None, sample: bool) -> None:  # noqa: FBT001
    tasks = cli.select_tasks()
    if len(tasks) > 1:
        utils.set_verbosity(utils.Verbosity.quiet)

    if solution is not None and len(tasks) > 1:
        utils.fatal_error(
            "A solution can only be specified when there's a single target task.",
        )

    solution_path = Path(solution) if solution else None

    status = utils.Status.success
    for task in tasks:
        status &= task.gen_expected(sample=sample, solution=solution_path)

    exit_with_status(status)


@cloup.command(help="Build statement pdf")
@cloup.pass_obj
def build_statement(cli: CLI) -> None:
    tasks = cli.select_tasks()

    for task in tasks:
        task.build_statement()


@cloup.command(help="Generate zip file with all test data")
@cloup.option(
    "--random-sort",
    "-r",
    is_flag=True,
    default=False,
    help="Add random prefix to output filenames to sort testcases within a subtask randomly.",
)
@cloup.pass_obj
def compress_dataset(cli: CLI, random_sort: bool) -> None:  # noqa: FBT001
    tasks = cli.select_tasks()

    for task in tasks:
        task.compress_dataset(random_sort=random_sort)


@cloup.command(help="Normalize input and output files running dos2unix")
@cloup.pass_obj
def normalize(cli: CLI) -> None:
    tasks = cli.select_tasks()

    for task in tasks:
        task.normalize()


@cloup.command(help="Run test plan")
@cloup.option("--subtask", "-st", type=int, help="Only run test plan for this subtask")
@cloup.pass_obj
def run_testplan(cli: CLI, subtask: int | None) -> None:
    tasks = cli.select_tasks()
    if len(tasks) > 1:
        utils.set_verbosity(utils.Verbosity.quiet)

    if subtask is not None and len(tasks) > 1:
        utils.fatal_error(
            "A subtask can only be specified when there's a single target task.",
        )
    status = utils.Status.success
    for task in tasks:
        status &= task.run_testplan(stn=Stn(subtask) if subtask else None)

    exit_with_status(status)


@cloup.command(help="Run input validators")
@cloup.option("--subtask", "-st", type=int, help="Only run validator for this subtask.")
@cloup.pass_obj
def validate_input(cli: CLI, subtask: int | None) -> None:
    tasks = cli.select_tasks()
    if len(tasks) > 1:
        utils.set_verbosity(utils.Verbosity.quiet)

    status = utils.Status.success
    for task in tasks:
        status &= task.validate_input(stn=Stn(subtask) if subtask else None)

    exit_with_status(status)


@cloup.command(help="Validate format of expected output files")
@cloup.option(
    "--subtask",
    "-st",
    type=int,
    help="Only validate output for this subtask.",
)
@cloup.pass_obj
def validate_output(cli: CLI, subtask: int | None) -> None:
    tasks = cli.select_tasks()
    if len(tasks) > 1:
        utils.set_verbosity(utils.Verbosity.quiet)

    status = utils.Status.success
    for task in tasks:
        status &= task.validate_output(stn=Stn(subtask) if subtask else None)

    exit_with_status(status)


@cloup.command(help="Print score parameters for cms")
@cloup.pass_obj
def score_params(cli: CLI) -> None:
    tasks = cli.select_tasks()

    for task in tasks:
        task.score_params()


@cloup.command(help="List all solutions")
@cloup.pass_obj
def list_solutions(cli: CLI) -> None:
    tasks = cli.select_tasks()

    for task in tasks:
        task.list_solutions()


def exit_with_status(status: utils.Status) -> NoReturn:
    match status:
        case utils.Status.success:
            sys.exit(0)
        case utils.Status.fail:
            sys.exit(2)


single_task = cloup.option(
    "--task",
    "task_name",
    help="Force command to run on the specified task instead of the one in the current directory",
)


@cloup.command(
    "run",
    short_help="Run a solution",
    help="Run a solution against all test data and display the output of the checker and running time",
)
@cloup.argument(
    "solution",
    help="A path to a solution. " + _SOLUTION_HELP,
    type=click.Path(),
    shell_complete=_solution_completion(),
)
@single_task
@mutually_exclusive(
    cloup.option(
        "--subtask",
        "-st",
        "stn",
        type=int,
        help="Only run solution on the given subtask",
    ),
    cloup.option(
        "--file",
        "-f",
        type=click.Path(),
        help="Run solution on the given file instead of the dataset. Use '-' to read from stdin.",
    ),
)
@cloup.option("--timeout", help="Timeout in seconds (default: 3.0)", type=float)
@cloup.constraint(
    If("file", then=accept_none).rephrased(
        error="--timeout cannot be used with --file",
    ),
    ["timeout"],
)
@cloup.pass_obj
def run_solution(
    cli: CLI,
    solution: str,
    task_name: str | None,
    stn: int | None,
    file: str | None,
    timeout: float | None,
) -> None:
    timeout = timeout or 3.0
    task = cli.select_task(task_name)
    if not task:
        utils.fatal_error("You have to be inside a task to run this command.")
    if file is not None:
        sol = task.load_solution_from_path(Path(solution))
        if not sol:
            return utils.show_message("Error", "Solution not found", utils.ERROR)
        sol.run_on_input(sys.stdin if file == "-" else Path(file))
    else:
        task.run_solution(
            Path(solution),
            timeout,
            stn=Stn(stn) if stn else None,
        )


@cloup.command(help="Build a solution")
@single_task
@cloup.argument(
    "solution",
    help="A path to a solution. " + _SOLUTION_HELP,
    type=click.Path(),
)
@cloup.pass_obj
def build(cli: CLI, solution: str, task_name: str | None) -> None:
    task = cli.select_task(task_name)
    if not task:
        utils.fatal_error("You have to be inside a task to run this command.")
    task.build_solution(Path(solution))


@cloup.command(
    short_help="Generate shell completion scripts",
    help="""
    Generate shell completion scripts for ocimatic commands.

    ### Bash

    First, install `bash-completion` using your package manager.

    \b
    Then, add this to your `~/.bash_profile`:
        eval "$(ocimatic completion bash)"

    ### Zsh

    \b
    Add this to ~/.zshrc:
        eval "$(ocimatic completion zsh)"

    ### Fish

    \b
    Generate an `ocimatic.fish` completion script:
        ocimatic completion fish > ~/.config/fish/completions/ocimatic.fish
    """,
)
@cloup.argument(
    "shell",
    type=click.Choice(["bash", "zsh", "fish"]),
)
def completion(shell: Literal["bash", "zsh", "fish"]) -> None:
    os.environ["_OCIMATIC_COMPLETE"] = f"{shell}_source"
    cli()


@cloup.command(
    short_help="Check if ocimatic is correctly setup",
    help="Check ocimatic is correctly setup by running some commands.",
)
def check_setup() -> None:
    utils.writeln("Running commands to check if they are available...", utils.INFO)
    utils.writeln()

    status = utils.Status.success
    status &= _test_command(ocimatic.config.python.command)
    utils.writeln()
    status &= _test_command(ocimatic.config.cpp.command)
    utils.writeln()
    status &= _test_command(ocimatic.config.java.javac)
    utils.writeln()
    status &= _test_command(ocimatic.config.java.jre)
    utils.writeln()
    status &= _test_command(ocimatic.config.rust.command)
    utils.writeln()
    status &= _test_command(ocimatic.config.latex.command)
    utils.writeln()

    if status == utils.Status.success:
        utils.writeln("All commands ran successfully.", utils.GREEN)
    else:
        utils.writeln(
            "Some commands failed to run. You can still try to use ocimatic\n"
            "but some solutions or generators may fail to run. You can use\n"
            "`ocimatic setup` to override the default configuration.",
            utils.RED,
        )

    exit_with_status(status)


def _test_command(cmd: str) -> utils.Status:
    try:
        utils.writeln(f"$ {cmd} --version", utils.INFO)
        subprocess.run([cmd, "--version"], check=True)
        return utils.Status.success
    except Exception as e:
        utils.writeln(f"command failed: {e}", utils.RED)
        return utils.Status.fail


@cloup.command(
    short_help="Setup ocimatic",
    help="Generate configuration file for ocimatic that can be used to override default commands.",
)
def setup() -> None:
    shutil.copy2(ocimatic.Config.DEFAULT_PATH, ocimatic.Config.HOME_PATH)

    utils.writeln(
        f"Configuration file created at '{ocimatic.Config.HOME_PATH}'.\n"
        "You can configure ocimatic by editing the file.",
        utils.OK,
    )


SECTIONS = [
    cloup.Section(
        "Contest commands",
        [
            init,
            new_task,
            problemset,
            archive,
            run_server,
        ],
    ),
    cloup.Section(
        "Multi-task commands",
        [
            run_testplan,
            gen_expected,
            validate_input,
            validate_output,
            check_dataset,
            build_statement,
            compress_dataset,
            normalize,
            score_params,
            list_solutions,
        ],
    ),
    cloup.Section(
        "Single-task commands",
        [
            run_solution,
            build,
        ],
    ),
    cloup.Section(
        "Config commands",
        [
            completion,
            check_setup,
            setup,
        ],
    ),
]


@cloup.group(
    help="""
A contest consists of a set of tasks. Ocimatic provides a set of commands that can work on multiple
tasks at the same time. We refer to the set of tasks a commands runs on as the list of *targets*.
To facilitate the selection of targets, Ocimatic is sensitive to the directory where you run it.
When inside a task's directory (or any subdiretory) that single task is selected as the target. When
run at the root of the contest, all tasks will be selected as targets.

Some commands are only valid if there's a single target task (Single-task commands). Some commands
apply to the entire contest (Contest commands) or are used to configure Ocimatic (Config commands)
and do not have a corresponding set of targets.
""",
    sections=SECTIONS,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@cloup.pass_context
def cli(ctx: click.Context) -> None:
    ctx.obj = CLI()
    # Only initialize config if we are not running the `setup` command. This ensures we can
    # run `ocimatic setup` even if there are issues with the config file.
    if ctx.invoked_subcommand != "setup":
        ocimatic.config.initialize()


def main() -> None:
    cli()
