from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ocimatic import ui
from ocimatic.runnable import ret_code_to_str
from ocimatic.source_code import BuildError, CppSource, PythonSource, SourceCode
from ocimatic.ui import Error

# https://en.wikipedia.org/wiki/C0_and_C1_control_codes#Field_separators
FS = chr(28)


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
                f'File not found: "{self._testplan_path}"',
            )
        self._task_directory = task_directory
        self._dataset_dir = dataset_directory

        subtasks = self._parse_file()
        if isinstance(subtasks, Error):
            ui.writeln(
                f"Error when parsing testplan in `{ui.relative_to_cwd(self._testplan_path)}`",
                ui.ERROR,
            )
            ui.writeln(subtasks.msg, ui.ERROR)
            sys.exit(1)

        self._subtasks = subtasks

    @property
    def subtasks(self) -> int:
        return len(self._subtasks)

    def validators(self) -> list[Path | None]:
        return [subtask.validator for subtask in self._subtasks]

    def run(self, stn: int | None) -> ui.Status:
        cwd = Path.cwd()
        # Run generators with `testplan/` as the cwd
        os.chdir(self._directory)

        status = ui.Status.success
        for i, st in enumerate(self._subtasks, 1):
            if stn is not None and stn != i:
                continue
            status &= st.run()

        if sum(len(st.commands) for st in self._subtasks) == 0:
            ui.show_message(
                "Warning",
                "no commands were executed for the plan.",
                ui.WARNING,
            )

        os.chdir(cwd)

        return status

    def _parse_file(self) -> list[_SubtaskPlan] | Error:
        subtasks: dict[int, _SubtaskPlan] = {}
        st = 0
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
                        return Error(
                            f"line {lineno}: found subtask {found_st}, but subtask {st + 1} was expected",
                        )
                    st += 1
                    subtasks[st] = _SubtaskPlan(self._dataset_dir, st, validator)
                elif cmd_match:
                    if st == 0:
                        return Error(
                            f"line {lineno}: found command before declaring a subtask.",
                        )
                    group = cmd_match.group(1)
                    cmd = cmd_match.group(2)
                    args = _parse_args(cmd_match.group(3) or "")

                    command = self._parse_command(
                        group,
                        cmd,
                        args,
                        lineno,
                    )
                    if isinstance(command, Error):
                        return command
                    subtasks[st].commands.append(command)
                else:
                    return Error(
                        f"line {lineno}: invalid line `{line}`\n",
                    )
        return [st for (_, st) in sorted(subtasks.items())]

    def _parse_command(
        self,
        group: str,
        cmd: str,
        args: list[str],
        lineno: int,
    ) -> _Command | Error:
        if cmd == "copy":
            if len(args) > 2:
                return Error(
                    f"line {lineno}: the `copy` command expects exactly one argument.",
                )
            return _Copy(group, self._task_directory, args[0])
        elif cmd == "echo":
            return _Echo(group, args)
        elif Path(cmd).suffix == ".py":
            return _Script(group, Path(self._directory, cmd), "py", args)
        elif Path(cmd).suffix == ".cpp":
            return _Script(group, Path(self._directory, cmd), "cpp", args)
        else:
            return _invalid_command_err(cmd, lineno)


@dataclass
class _Include:
    stn: int
    patterh: str


class _SubtaskPlan:
    def __init__(self, dataset_dir: Path, stn: int, validator: Path | None) -> None:
        self._dir = Path(dataset_dir, f"st{stn}")
        self.commands: list[_Command] = []
        self.validator = validator

    def __str__(self) -> str:
        return str(self._dir.name)

    @ui.workgroup("{0}")
    def run(self) -> ui.Status:
        shutil.rmtree(self._dir, ignore_errors=True)
        self._dir.mkdir(parents=True, exist_ok=True)

        status = ui.Status.success
        tests_in_group: Counter[str] = Counter()
        for cmd in self.commands:
            status &= cmd.run(self._dir, tests_in_group).status
        return status


class _Command(ABC):
    def __init__(self, group: str) -> None:
        self._group = group

    def dst_file(self, directory: Path, idx: int) -> Path:
        return Path(directory, f"{self._group}-{idx}.in")

    @abstractmethod
    def run(self, dst_dir: Path, tests_in_group: Counter[str]) -> ui.Result:
        raise NotImplementedError(
            f"Class {self.__class__.__name__} doesn't implement run()",
        )


class _Copy(_Command):
    magic_check = re.compile("([*?[])")

    def __init__(self, group: str, dir: Path, pattern: str) -> None:
        super().__init__(group)
        self._dir = dir
        self._pattern = pattern

    def __str__(self) -> str:
        return self._pattern

    @ui.work("Copy", "{0}")
    def run(self, dst_dir: Path, tests_in_group: Counter[str]) -> ui.Result:
        files = list(self._dir.glob(self._pattern))
        if not files:
            msg = "No file matches the pattern" if self.has_magic() else "No such file"
            return ui.Result.fail(short_msg=msg)
        try:
            for file in files:
                idx = _next_idx_in_group(self._group, tests_in_group)
                shutil.copy(file, self.dst_file(dst_dir, idx))
            return _success_with_count_result(len(files))
        except Exception:  # pylint: disable=broad-except
            return ui.Result.fail(short_msg="Error when copying file")

    def has_magic(self) -> bool:
        return _Copy.magic_check.search(self._pattern) is not None


class _Echo(_Command):
    def __init__(self, group: str, args: list[str]) -> None:
        super().__init__(group)
        self._args = args

    def __str__(self) -> str:
        return str(self._args)

    @ui.work("Echo", "{0}")
    def run(self, dst_dir: Path, tests_in_group: Counter[str]) -> ui.Result:
        idx = _next_idx_in_group(self._group, tests_in_group)
        with self.dst_file(dst_dir, idx).open("w") as test_file:
            test_file.write(" ".join(self._args) + "\n")
            return _success_with_count_result(1)


class _Script(_Command):
    VALID_EXTENSIONS = Literal["py", "cpp"]

    def __init__(
        self,
        group: str,
        path: Path,
        ext: VALID_EXTENSIONS,
        args: list[str],
    ) -> None:
        super().__init__(group)
        self._args = args
        self._script_path = path
        self._ext = ext

    def __str__(self) -> str:
        args = " ".join(self._args)
        script = self._script_path.name
        return f"{self._group} ; {script} {args}"

    @ui.work("Gen", "{0}")
    def run(self, dst_dir: Path, tests_in_group: Counter[str]) -> ui.Result:
        script = self._load_script()
        if not script:
            return ui.Result.fail(short_msg="Script file not found")
        build_result = script.build()
        if isinstance(build_result, BuildError):
            return ui.Result.fail(
                short_msg="Failed to build generator",
                long_msg=build_result.msg,
            )

        count = 0
        idx = _next_idx_in_group(self._group, tests_in_group)
        # We seed the script with the first `idx`, this guarantees it will be different next time.
        process = build_result.spawn([self._seed_arg(dst_dir, idx), *self._args])
        current_file = None
        try:
            assert process.stdout
            while char := process.stdout.read(1):
                if char == FS:
                    if current_file:
                        current_file.close()
                        current_file = None
                else:
                    if current_file is None:
                        count += 1
                        current_file = self.dst_file(dst_dir, idx).open("w")
                        idx = _next_idx_in_group(self._group, tests_in_group)
                    current_file.write(char)
        finally:
            if current_file:
                current_file.close()

        if count == 0:
            return ui.Result.fail(short_msg="generator didn't produce any output")

        ret = process.wait()
        if ret == 0:
            return _success_with_count_result(count)
        else:
            msg = ret_code_to_str(ret)
            long_msg = process.stderr.read() if process.stderr else None
            return ui.Result.fail(short_msg=msg, long_msg=long_msg)

    def _load_script(self) -> SourceCode | None:
        if not self._script_path.exists():
            return None
        if self._ext == "py":
            return PythonSource(self._script_path)
        elif self._ext == "cpp":
            return CppSource(self._script_path)

    def _seed_arg(self, directory: Path, idx: int) -> str:
        return f"{directory.name}-{self._group}-{idx}"


def _invalid_command_err(cmd: str, lineno: int) -> Error:
    from typing import get_args

    extensions = get_args(_Script.VALID_EXTENSIONS)
    return Error(
        f"line {lineno}: invalid command `{cmd}`\n"
        f"The command should be either `copy`, `echo` or a generator with one of the following extensions {extensions}",
    )


def _success_with_count_result(count: int) -> ui.Result:
    assert count > 0
    if count == 1:
        return ui.Result.success(short_msg="1 test case generated")
    else:
        return ui.Result.success(short_msg=f"{count} test cases generated")


def _parse_args(args: str) -> list[str]:
    args = args.strip()
    return [a.encode().decode("unicode_escape") for a in shlex.split(args)]


def _next_idx_in_group(group: str, tests_in_group: Counter[str]) -> int:
    tests_in_group[group] += 1
    return tests_in_group[group]
