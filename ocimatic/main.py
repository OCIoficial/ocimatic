from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal

import click
import cloup
from click.shell_completion import CompletionItem
from cloup.constraints import If, accept_none, mutually_exclusive

# Do not import anything unnecessary to speed up loading. This makes a noticeable difference
# when cloup is computing completions.
if TYPE_CHECKING:
    from collections.abc import Callable

    from cloup.typing import Decorator
    from ocimatic.core import Task
    from ocimatic.result import Status
    from ocimatic.utils import Stn


_SOLUTION_HELP = (
    "If the path is absolute, load solution directly from the path. "
    "If the path is relative, try finding the solution relative to the following locations "
    "until a match is found (or we fail to find one): '<task>/solutions/correct', "
    "'<task>/solutions/partial', '<task>/solutions/', and '<cwd>'. "
    "Here, <task> refers to the path of the current task and <cwd> to the current working directory."
)


def _exits_with_status[**P](f: Callable[P, Status]) -> Callable[P, None]:
    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        import sys
        from ocimatic.result import Status

        match f(*args, **kwargs):
            case Status.success:
                sys.exit(0)
            case Status.fail:
                sys.exit(2)

    return wrapper


def _solution_completion(
    *,
    partial: bool = True,
) -> Callable[[click.Context, click.Parameter, str], list[CompletionItem]]:
    def inner(
        ctx: click.Context,
        param: click.Parameter,
        incomplete: str,
    ) -> list[CompletionItem]:
        from pathlib import Path

        from ocimatic.config import Config
        from ocimatic.env import Env
        from ocimatic.core import Contest, current_task_dir, find_contest_root

        try:
            del param
            cwd = Path.cwd()
            root = find_contest_root(cwd)
            if root is None:
                return []

            # Completion never builds or runs anything, so the default configuration is enough.
            with Env.use(Env(config=Config(), cwd=cwd, contest_root=root)):
                task_name: str | None = ctx.params.get("task_name")

                task = None
                if task_name is not None:
                    task = Contest.load_task_by_name(root, task_name)
                elif (task_dir := current_task_dir()) is not None:
                    task = Contest.load_task_by_dir(root, task_dir)

                if not task:
                    return []

                return task.solution_completion(incomplete, partial=partial)
        except Exception:
            return []

    return inner


def _subtask_option(*, help: str) -> Decorator:
    """Declare the `--subtask` option, passed to the command as `subtask`."""
    return cloup.option(
        "--subtask",
        "-st",
        "subtask",
        type=cloup.IntRange(min=1),
        help=help,
    )


def _check_subtask(tasks: list[Task], subtask: int | None) -> Stn | None:
    """Check the `--subtask` option against the target tasks and return it as a `Stn`."""
    from ocimatic.errors import OcimaticError
    from ocimatic.utils import Stn

    if subtask is None:
        return None
    if len(tasks) > 1:
        raise OcimaticError(
            "A subtask can only be specified when there's a single target task.",
        )
    [task] = tasks
    stn = Stn(subtask)
    if stn not in task.subtasks():
        count = len(task.subtasks())
        raise OcimaticError(
            f"Subtask {subtask} doesn't exist: `{task}` has {count} "
            f"{'subtask' if count == 1 else 'subtasks'}.",
        )
    return stn


@cloup.command(help="Initialize a contest in a new directory.")
@cloup.argument("path", help="Path to directory.")
@cloup.option("--phase", help="The contest phase used in the generated pdfs.")
@cloup.option(
    "--typesetting",
    type=cloup.Choice(["typst", "latex"]),
    help="Software used for typesetting documents.",
)
def init(path: str, phase: str | None, typesetting: str | None) -> None:
    from pathlib import Path

    from ocimatic import ui
    from ocimatic.errors import OcimaticError
    from ocimatic.core import Contest, Typesetting
    import questionary

    # Ask for phase if not provided
    if not phase:
        phase = questionary.select(
            "Select contest phase:",
            choices=[
                "Regional",
                "Final Nacional",
                "Other (enter manually)",
            ],
        ).ask()
        if phase == "Other (enter manually)":
            phase = questionary.text("Enter contest phase:").ask()

        if phase is None:
            return

    # Ask for typesetting if not provided
    if not typesetting:
        typesetting = questionary.select(
            "Select typesetting software:",
            choices=["typst", "latex"],
        ).ask()
        if typesetting is None:
            return

    try:
        ui.writeln()
        contest_path = Path(Path.cwd(), path)
        if contest_path.exists():
            raise OcimaticError("Couldn't create contest. Path already exists")
        Contest.create_layout(contest_path, phase, Typesetting(typesetting))
        ui.show_message("Info", f"Contest [{path}] created", ui.OK)
    except OcimaticError:
        raise
    except Exception as exc:
        raise OcimaticError("Couldn't create contest.", details=str(exc)) from exc


@cloup.command(
    "server",
    short_help="Start a server to control Ocimatic from a browser.",
    help="Start a server which can be used to control ocimatic from a browser. "
    "This is currently very limited, but it's useful during a contest to quickly paste "
    "and run a solution.",
)
@cloup.option("--port", "-p", default="9999", type=int)
def run_server(port: int) -> None:
    import sys
    from pathlib import Path

    from ocimatic import core, server

    server.run(Path(sys.argv[0]), core.Contest.load(), port)


@cloup.command(
    "sync-resources",
    short_help="Copy resources (e.g., `oci.typ`) into the contest.",
    help="Copy resources (e.g., `oci.typ`) into the contest and tasks. This is useful for fixing bugs "
    "in Ocimatic after a contest or a task has already been initialized. This is a risky operation. "
    "Don't use it unless you know what you are doing.",
)
def sync_resources() -> None:
    import questionary
    from ocimatic import core, ui

    ui.writeln(
        "This will overwrite resource files in the contest and all tasks.",
        ui.INFO,
    )
    answer = questionary.confirm(
        "Are you sure you want to continue?",
        default=False,
    ).ask()
    if answer is not True:
        return
    core.Contest.load().sync_resources()


@cloup.command(help="Generate the problemset PDF.")
@_exits_with_status
def problemset() -> Status:
    from ocimatic import core

    return core.Contest.load().build_problemset()


@cloup.command(
    short_help="Create a zip archive of the contest.",
    help="Create a zip archive of the contest containing the statements and dataset.",
)
@_exits_with_status
def archive() -> Status:
    from ocimatic import core

    return core.Contest.load().archive()


def _validate_task_name(ctx: click.Context, param: click.Argument, value: str) -> str:
    del ctx, param
    if not value.isalpha():
        raise click.BadParameter("Task name must be a word containing only letters.")
    return value


@cloup.command(help="Create a new task.")
@cloup.argument("name", help="Name of the task.", callback=_validate_task_name)
def new_task(name: str) -> None:
    from ocimatic import core

    core.new_task(core.Contest.load(), name)


@cloup.command(
    short_help="Run dataset validation.",
    help="""
Runs multiple validations on the dataset:\n
 - Check input/output correctness by running all correct solutions against all test cases.\n
 - Test robustness by running partial solutions and verifying they fail the subtasks they are suppose to.\n
 - Validate input/output formatting, ensuring basic rules like lines not ending with spaces.\n
 - Run all input validators.\n
""",
)
@_exits_with_status
def check_dataset() -> Status:
    from ocimatic import core, ui
    from ocimatic.env import Env, Verbosity
    from ocimatic.result import Status

    with Env.override(verbosity=Verbosity.quiet):
        tasks = core.select_tasks(core.Contest.load())
        failed = [task for task in tasks if task.check_dataset() == Status.fail]
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

        return Status.fail if failed else Status.success


@cloup.command(
    short_help="Generate expected output.",
    help="Generate expected output by running a correct solution against all input data. "
    "By default, it will choose any correct solution, preferring those "
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
    help="Generate expected output for sample input as well.",
    is_flag=True,
    default=False,
)
@_exits_with_status
def gen_expected(solution: str | None, sample: bool) -> Status:  # noqa: FBT001
    from pathlib import Path

    from ocimatic import core, ui
    from ocimatic.env import Env, Verbosity
    from ocimatic.errors import OcimaticError
    from ocimatic.result import Status

    tasks = core.select_tasks(core.Contest.load())
    with Env.override(
        verbosity=Verbosity.quiet if len(tasks) > 1 else Verbosity.verbose,
    ):
        if solution is not None and len(tasks) > 1:
            raise OcimaticError(
                "A solution can only be specified when there's a single target task.",
            )

        solution_path = Path(solution) if solution else None

        failed = [
            task
            for task in tasks
            if task.gen_expected(sample=sample, solution=solution_path) == Status.fail
        ]

        if len(tasks) > 1 and len(failed) > 0:
            ui.writeln(
                """
--------------------------------------------------------
Failed to generate expected output for some of the tasks.

Tasks with issues:""",
                ui.ERROR,
            )
            for t in failed:
                ui.writeln(f" * {t}", ui.ERROR)
            ui.writeln(
                """
To investigate further, run `ocimatic gen-expected`
inside the corresponding task directory to get detailed
information about the failures.
--------------------------------------------------------
""",
                ui.ERROR,
            )
        return Status.fail if failed else Status.success


@cloup.command(help="Build the statement PDF.")
@_exits_with_status
def build_statement() -> Status:
    from ocimatic import core
    from ocimatic.result import Status

    status = Status.success
    for task in core.select_tasks(core.Contest.load()):
        status &= task.build_statement()
    return status


@cloup.command(help="Generate zip file with all test data.")
@cloup.option(
    "--random-sort",
    "-r",
    is_flag=True,
    default=False,
    help="Add random prefix to output filenames to randomly sort testcases within a subtask",
)
@_exits_with_status
def compress_dataset(random_sort: bool) -> Status:  # noqa: FBT001
    from ocimatic import core
    from ocimatic.result import Status

    status = Status.success
    for task in core.select_tasks(core.Contest.load()):
        status &= task.compress_dataset(random_sort=random_sort)
    return status


@cloup.command(help="Normalize input and output files running dos2unix.")
@_exits_with_status
def normalize() -> Status:
    from ocimatic import core
    from ocimatic.result import Status

    status = Status.success
    for task in core.select_tasks(core.Contest.load()):
        status &= task.normalize()
    return status


@cloup.command(help="Run the test plan.")
@_subtask_option(
    help="Only run the test plan for this subtask. "
    " This option can only be specified if there's a single target task.",
)
@cloup.option(
    "--gen-expected",
    is_flag=True,
    default=False,
    help="Generate expected output after running testplan.",
)
@_exits_with_status
def run_testplan(
    subtask: int | None,
    gen_expected: bool,  # noqa: FBT001
) -> Status:
    from ocimatic import core, ui
    from ocimatic.env import Env, Verbosity
    from ocimatic.result import Status

    tasks = core.select_tasks(core.Contest.load())
    with Env.override(
        verbosity=Verbosity.quiet if len(tasks) > 1 else Verbosity.verbose,
    ):
        stn = _check_subtask(tasks, subtask)
        failed = [task for task in tasks if task.run_testplan(stn=stn) == Status.fail]
        if len(failed) == 0 and gen_expected:
            failed = [
                task for task in tasks if task.gen_expected(stn=stn) == Status.fail
            ]

        if len(tasks) > 1 and len(failed) > 0:
            ui.writeln(
                """
----------------------------------------------------
Testplan failed for some of the tasks.

Tasks with issues:""",
                ui.ERROR,
            )
            for t in failed:
                ui.writeln(f" * {t}", ui.ERROR)
            ui.writeln(
                """
To investigate further, run `ocimatic run-testplan`
inside the corresponding task directory to get
detailed information about the failures.
----------------------------------------------------
""",
                ui.ERROR,
            )
        return Status.fail if failed else Status.success


@cloup.command(help="Run input validators.")
@_subtask_option(help="Only run validator for this subtask.")
@_exits_with_status
def validate_input(subtask: int | None) -> Status:
    from ocimatic import core
    from ocimatic.env import Env, Verbosity
    from ocimatic.result import Status

    tasks = core.select_tasks(core.Contest.load())
    with Env.override(
        verbosity=Verbosity.quiet if len(tasks) > 1 else Verbosity.verbose,
    ):
        stn = _check_subtask(tasks, subtask)
        status = Status.success
        for task in tasks:
            status &= task.validate_input(stn=stn)

        return status


@cloup.command(help="Validate the format of expected output files.")
@_subtask_option(help="Only validate output for this subtask.")
@_exits_with_status
def validate_output(subtask: int | None) -> Status:
    from ocimatic import core
    from ocimatic.env import Env, Verbosity
    from ocimatic.result import Status

    tasks = core.select_tasks(core.Contest.load())
    with Env.override(
        verbosity=Verbosity.quiet if len(tasks) > 1 else Verbosity.verbose,
    ):
        stn = _check_subtask(tasks, subtask)
        status = Status.success
        for task in tasks:
            status &= task.validate_output(stn=stn)

        return status


@cloup.command(help="Print score parameters for CMS.")
@_exits_with_status
def score_params() -> Status:
    from ocimatic import core
    from ocimatic.result import Status

    status = Status.success
    for task in core.select_tasks(core.Contest.load()):
        status &= task.score_params()
    return status


@cloup.command(help="List all solutions.")
def list_solutions() -> None:
    from ocimatic import core

    tasks = core.select_tasks(core.Contest.load())

    for task in tasks:
        task.list_solutions()


@cloup.command(help="Compute code coverage.")
def coverage() -> None:
    from ocimatic import core

    tasks = core.select_tasks(core.Contest.load())

    for task in tasks:
        task.coverage()


single_task = cloup.option(
    "--task",
    "task_name",
    help="Force command to run on the specified task instead of the one in the current directory.",
)


@cloup.command(
    "run",
    short_help="Run a solution.",
    help="Run a solution against all test data and display the output of the checker and running time.",
)
@cloup.argument(
    "solution",
    help="A path to a solution. " + _SOLUTION_HELP,
    type=click.Path(),
    shell_complete=_solution_completion(),
)
@single_task
@mutually_exclusive(
    _subtask_option(help="Only run solution on the given subtask."),
    cloup.option(
        "--file",
        "-f",
        type=click.Path(),
        help="Run solution on the given file instead of the dataset. Use '-' to read from stdin.",
    ),
)
@cloup.option("--timeout", help="Timeout in seconds (default: 3.0).", type=float)
@cloup.constraint(
    If("file", then=accept_none).rephrased(
        error="--timeout cannot be used with --file",
    ),
    ["timeout"],
)
@_exits_with_status
def run_solution(
    solution: str,
    task_name: str | None,
    subtask: int | None,
    file: str | None,
    timeout: float | None,
) -> Status:
    import sys
    from pathlib import Path

    from ocimatic import core, ui
    from ocimatic.errors import OcimaticError
    from ocimatic.result import Status

    task = core.select_task(core.Contest.load(), task_name)
    if not task:
        raise OcimaticError("You have to be inside a task to run this command.")
    stn = _check_subtask([task], subtask)
    if file is not None:
        sol = task.load_solution_from_path(Path(solution))
        if not sol:
            ui.show_message("Error", "Solution not found", ui.ERROR)
            return Status.fail
        return sol.run_on_input(sys.stdin if file == "-" else Path(file))
    return task.run_solution(
        Path(solution),
        timeout=timeout or 3.0,
        stn=stn,
    )


@cloup.command(help="Build a solution.")
@single_task
@cloup.argument(
    "solution",
    help="A path to a solution. " + _SOLUTION_HELP,
    type=click.Path(),
)
@_exits_with_status
def build(solution: str, task_name: str | None) -> Status:
    from pathlib import Path

    from ocimatic import core
    from ocimatic.errors import OcimaticError

    task = core.select_task(core.Contest.load(), task_name)
    if not task:
        raise OcimaticError("You have to be inside a task to run this command.")
    return task.build_solution(Path(solution))


@cloup.command(
    short_help="Generate shell completion scripts.",
    help="""
    Generate shell completion scripts for Ocimatic.

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
    import os

    os.environ["_OCIMATIC_COMPLETE"] = f"{shell}_source"
    cli()


@cloup.command(
    short_help="Check if Ocimatic is correctly setup.",
    help="Check Ocimatic is correctly setup by running some commands.",
)
@_exits_with_status
def check_setup() -> Status:
    import tempfile
    from pathlib import Path

    from ocimatic import ui
    from ocimatic.result import Status
    from ocimatic.source_code import (
        CppSource,
        JavaSource,
        LatexSource,
        PythonSource,
        RustSource,
    )

    ui.writeln("Running commands to check if they are available...", ui.INFO)

    resources = Path(__file__).parent / "resources" / "tests"

    status = Status.success
    with tempfile.TemporaryDirectory() as tmp:
        status &= JavaSource.test(resources, Path(tmp))
        ui.writeln()
        status &= PythonSource.test(resources, Path(tmp))
        ui.writeln()
        status &= CppSource.test(resources, Path(tmp))
        ui.writeln()
        status &= RustSource.test(resources, Path(tmp))
        ui.writeln()
        status &= LatexSource.test(resources, Path(tmp))
        ui.writeln()

    if status == Status.success:
        ui.writeln("All commands ran successfully.", ui.GREEN)
    else:
        ui.writeln(
            "----------------------------------------------------------",
            ui.ERROR,
        )
        ui.writeln(
            "Some commands failed. You can still try to use ocimatic\n"
            "but some solutions or generators may not work. You can use\n"
            "`ocimatic setup` to override the default configuration.",
            ui.RED,
        )
        ui.writeln(
            "----------------------------------------------------------",
            ui.ERROR,
        )

    return status


@cloup.command(
    short_help="Setup Ocimatic.",
    help="Generate configuration file for Ocimatic that can be used to override default commands.",
)
def setup() -> None:
    from ocimatic import ui
    from ocimatic.config import Config
    import questionary
    import tomlkit

    if Config.HOME_PATH.exists():
        ui.writeln(
            f"Configuration file already exists at '{Config.HOME_PATH}'.\n"
            "This will overwrite the existing file.",
            ui.INFO,
        )
        answer = questionary.confirm(
            "Are you sure you want to continue?",
            default=False,
        ).ask()
        if answer is not True:
            return
        ui.writeln()

    doc = Config.default_toml_document()
    with Config.HOME_PATH.open("w") as f:
        tomlkit.dump(doc, f)  # pyright: ignore[reportUnknownMemberType]

    ui.writeln(
        f"Configuration file created at '{Config.HOME_PATH}'.\n"
        "You can configure ocimatic by editing the file.",
        ui.INFO,
    )


@cloup.command(
    help="Show version and exit.",
)
def version() -> None:
    from ocimatic._version import __version__

    print(__version__)


@cloup.command(
    help="Run lsp server.",
)
@mutually_exclusive(
    cloup.option("--tcp", help="start a TCP server", is_flag=True, default=False),
    cloup.option("--ws", help="start a WebSocket server", is_flag=True, default=False),
)
@cloup.option(
    "--host",
    default="127.0.0.1",
    help="bind to this address",
    show_default=True,
)
@cloup.option(
    "--port",
    type=int,
    default=8888,
    help="bind to this port",
    show_default=True,
)
def lsp(tcp: bool, ws: bool, host: str, port: int) -> None:  # noqa: FBT001
    from ocimatic.lsp import server

    if tcp:
        server.start_tcp(host, port)
    elif ws:
        server.start_ws(host, port)
    else:
        server.start_io()


SECTIONS = [
    cloup.Section(
        "Contest commands",
        [
            init,
            new_task,
            problemset,
            archive,
            run_server,
            sync_resources,
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
            coverage,
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
            version,
            completion,
            check_setup,
            setup,
            lsp,
        ],
    ),
]


@cloup.group(
    help="""
A contest consists of a set of tasks. Ocimatic provides a set of commands that can work on multiple
tasks at the same time. We refer to the set of tasks a command runs on as the list of *targets*.
To facilitate the selection of targets, ocimatic is sensitive to the directory where you run it.
When inside a task's directory (or any subdirectory), that single task is selected as the target. When
running ocimatic at the root of the contest, all tasks will be selected as targets. For some commands,
you can override the default set of targets by passing the --task flag.

Some commands are only valid if there's a single target task (Single-task commands). Some commands
apply to the entire contest (Contest commands) or are used to configure Ocimatic (Config commands),
and do not have a corresponding set of targets.

You can see more information about a command by calling it with --help/-h.
""",
    sections=SECTIONS,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@cloup.pass_context
def cli(ctx: click.Context) -> None:
    from pathlib import Path

    from ocimatic.config import Config
    from ocimatic.env import Env
    from ocimatic.core import find_contest_root

    # Don't load the config file for `setup`. This ensures we can run `ocimatic setup` even if
    # there are issues with the config file.
    config = Config() if ctx.invoked_subcommand == "setup" else Config.load()
    cwd = Path.cwd()
    ctx.with_resource(
        Env.use(
            Env(config=config, cwd=cwd, contest_root=find_contest_root(cwd)),
        ),
    )
