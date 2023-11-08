import os
import re
import shlex
import shutil
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Literal, NoReturn

import ocimatic
from ocimatic import ui
from ocimatic.runnable import RunError, RunSuccess
from ocimatic.source_code import BuildError, CppSource, PythonSource, SourceCode
from ocimatic.ui import Status, WorkResult


class Testplan:
    """Functionality to read and run a plan for generating dataset."""

    def __init__(
        self,
        directory: Path,
        task_directory: Path,
        dataset_directory: Path,
        filename: str = "testplan.txt",
    ) -> None:
        self._directory = directory
        self._testplan_path = Path(directory, filename)
        if not self._testplan_path.exists():
            ui.fatal_error(
                'No such file plan for creating dataset: "%s"' % self._testplan_path,
            )
        self._task_directory = task_directory
        self._dataset_dir = dataset_directory

    def validators(self) -> list[Path | None]:
        return [subtask.validator for subtask in self._parse_file()]

    def run(self, stn: int | None) -> Literal[ui.Status.success, ui.Status.fail]:
        subtasks = self._parse_file()
        cwd = Path.cwd()
        # Run generators with testplan/ as the cwd
        os.chdir(self._directory)

        status: Literal[ui.Status.success, ui.Status.fail] = ui.Status.success
        for i, st in enumerate(subtasks, 1):
            if stn is not None and stn != i:
                continue

            if st.run() is not ui.Status.success:
                status = ui.Status.fail

        if sum(len(st.commands) for st in subtasks) == 0:
            ui.show_message(
                "Warning",
                "no commands were executed for the plan.",
                ui.WARNING,
            )

        os.chdir(cwd)

        return status

    def _parse_file(self) -> list["Subtask"]:
        subtasks: dict[int, "Subtask"] = {}
        st = 0
        tests_in_group: Counter[str] = Counter()
        for lineno, line in enumerate(self._testplan_path.open("r").readlines(), 1):
            line = line.strip()
            subtask_header = re.compile(
                r"\s*\[\s*Subtask\s*(\d+)\s*(?:-\s*([^\]\s]+))?\s*\]\s*",
            )
            cmd_line = re.compile(r"\s*([^;\s]+)\s*;\s*(\S+)(:?\s+(.*))?")
            comment = re.compile(r"\s*#.*")

            if not line:
                continue
            if not comment.fullmatch(line):
                header_match = subtask_header.fullmatch(line)
                cmd_match = cmd_line.fullmatch(line)
                if header_match:
                    found_st = int(header_match.group(1))
                    validator = (
                        Path(self._directory, header_match.group(2))
                        if header_match.group(2)
                        else None
                    )
                    if st + 1 != found_st:
                        ui.fatal_error(
                            f"line {lineno}: found subtask {found_st}, but subtask {st + 1} was expected",
                        )
                    st += 1
                    subtasks[st] = Subtask(self._dataset_dir, st, validator)
                    tests_in_group = Counter()
                elif cmd_match:
                    if st == 0:
                        ui.fatal_error(
                            f"line {lineno}: found command before declaring a subtask.",
                        )
                    group = cmd_match.group(1)
                    cmd = cmd_match.group(2)
                    args = _parse_args(cmd_match.group(3) or "")

                    tests_in_group[group] += 1
                    command = self._parse_command(
                        group,
                        tests_in_group[group],
                        cmd,
                        args,
                        lineno,
                    )
                    subtasks[st].commands.append(command)
                else:
                    ui.fatal_error(
                        f"line {lineno}: error while parsing line `{line}`\n",
                    )
        return [st for (_, st) in sorted(subtasks.items())]

    def _parse_command(
        self,
        group: str,
        idx: int,
        cmd: str,
        args: list[str],
        lineno: int,
    ) -> "Command":
        if cmd == "copy":
            if len(args) > 2:
                ui.fatal_error(
                    f"line {lineno}: command `copy` expects exactly one argument.",
                )
            return Copy(group, idx, Path(self._task_directory, args[0]))
        elif cmd == "echo":
            return Echo(group, idx, args)
        elif Path(cmd).suffix == ".py":
            return Script(group, idx, Path(self._directory, cmd), "py", args)
        elif Path(cmd).suffix == ".cpp":
            return Script(group, idx, Path(self._directory, cmd), "cpp", args)
        else:
            _invalid_command(cmd, lineno)


class Subtask:
    def __init__(self, dataset_dir: Path, stn: int, validator: Path | None) -> None:
        self._dir = Path(dataset_dir, f"st{stn}")
        self.commands: list["Command"] = []
        self.validator = validator

    def __str__(self) -> str:
        return str(self._dir.name)

    @ui.workgroup("{0}")
    def run(self) -> Literal[ui.Status.success, ui.Status.fail]:
        shutil.rmtree(self._dir, ignore_errors=True)
        self._dir.mkdir(parents=True, exist_ok=True)

        status: Literal[ui.Status.success, ui.Status.fail] = ui.Status.success
        for cmd in self.commands:
            if cmd.run(self._dir).status is not Status.success:
                status = ui.Status.fail
        return status


class Command(ABC):
    def __init__(self, group: str, idx: int) -> None:
        self._group = group
        self._idx = idx

    def dst_file(self, directory: Path) -> Path:
        return Path(directory, f"{self._group}-{self._idx}.in")

    @abstractmethod
    def run(self, dst_dir: Path) -> WorkResult:
        raise NotImplementedError(
            f"Class {self.__class__.__name__} doesn't implement run()",
        )


class Copy(Command):
    def __init__(self, group: str, idx: int, file: Path) -> None:
        super().__init__(group, idx)
        self._file = file

    def __str__(self) -> str:
        return str(self._file.relative_to(ocimatic.config["contest_root"]))

    @ui.work("Copy", "{0}")
    def run(self, dst_dir: Path) -> WorkResult:
        if not self._file.exists():
            return WorkResult.fail(short_msg="No such file")
        try:
            shutil.copy(self._file, self.dst_file(dst_dir))
            return WorkResult.success(short_msg="OK")
        except Exception:  # pylint: disable=broad-except
            return WorkResult.fail(short_msg="Error when copying file")


class Echo(Command):
    def __init__(self, group: str, idx: int, args: list[str]) -> None:
        super().__init__(group, idx)
        self._args = args

    def __str__(self) -> str:
        return str(self._args)

    @ui.work("Echo", "{0}")
    def run(self, dst_dir: Path) -> WorkResult:
        with self.dst_file(dst_dir).open("w") as test_file:
            test_file.write(" ".join(self._args) + "\n")
            return WorkResult.success(short_msg="Ok")


class Script(Command):
    VALID_EXTENSIONS = Literal["py", "cpp"]

    def __init__(
        self,
        group: str,
        idx: int,
        path: Path,
        ext: VALID_EXTENSIONS,
        args: list[str],
    ) -> None:
        super().__init__(group, idx)
        self._args = args
        self._script_path = path
        self._ext = ext

    def __str__(self) -> str:
        args = " ".join(self._args)
        script = self._script_path.name
        return f"{self._group} ; {script} {args}"

    @ui.work("Gen", "{0}")
    def run(self, dst_dir: Path) -> WorkResult:
        script = self._load_script()
        if not script:
            return WorkResult.fail(short_msg="Script file not found")
        build_result = script.build()
        if isinstance(build_result, BuildError):
            return WorkResult.fail(
                short_msg="Failed to build generator",
                long_msg=build_result.msg,
            )
        result = build_result.run(
            out_path=self.dst_file(dst_dir),
            args=[self._seed_arg(dst_dir), *self._args],
        )
        match result:
            case RunSuccess(_):
                return WorkResult.success(short_msg="OK")
            case RunError(msg, stderr):
                return WorkResult.fail(short_msg=msg, long_msg=stderr)

    def _load_script(self) -> SourceCode | None:
        if not self._script_path.exists():
            return None
        if self._ext == "py":
            return PythonSource(self._script_path)
        elif self._ext == "cpp":
            return CppSource(self._script_path)

    def _seed_arg(self, directory: Path) -> str:
        return f"{directory.name}-{self._group}-{self._idx}"


def _invalid_command(cmd: str, lineno: int) -> NoReturn:
    from typing import get_args

    extensions = get_args(Script.VALID_EXTENSIONS)
    ui.fatal_error(
        f"line {lineno}: invalid command `{cmd}`\n"
        f"The command should be either `copy`, `echo` or a generator with one of the following extensions {extensions}",
    )


def _parse_args(args: str) -> list[str]:
    args = args.strip()
    return [a.encode().decode("unicode_escape") for a in shlex.split(args)]
