from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import typst

from ocimatic import ui, utils
from ocimatic.config import Config
from ocimatic.result import Error, Status, Result
from ocimatic.runnable import (
    Binary,
    JavaClasses,
    Python3,
    RunError,
    Runnable,
    RunSuccess,
)


@dataclass
class BuildError:
    msg: str


class SourceCode(ABC):
    LINE_COMMENT_START: str

    def __init__(self, file: Path) -> None:
        relative_path = utils.relative_to_cwd(file)
        self._file = file
        self.name = str(relative_path)

    def __str__(self) -> str:
        return self.name

    @property
    def file(self) -> Path:
        return self._file

    @abstractmethod
    def build(
        self,
        *,
        force: bool = False,
    ) -> Runnable | BuildError: ...

    def comments_iter(self) -> Iterator[str]:
        comment_start = self.__class__.LINE_COMMENT_START
        with self._file.open() as file:
            for line in file:
                if line.startswith(comment_start):
                    yield line.removeprefix(comment_start)


class CompiledSource(SourceCode):
    @abstractmethod
    def _build_cmd(self) -> list[str]: ...

    @abstractmethod
    def _runnable(self) -> Runnable: ...

    @abstractmethod
    def _should_build(self) -> bool: ...

    @abstractmethod
    def _ensure_out_dir(self) -> None: ...

    def build(
        self,
        *,
        force: bool = False,
    ) -> Runnable | BuildError:
        if force or self._should_build():
            self._ensure_out_dir()
            try:
                complete = subprocess.run(
                    self._build_cmd(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if complete.returncode != 0:
                    return BuildError(msg=complete.stderr)
            except Exception as e:
                return BuildError(msg=str(e))
        return self._runnable()


class CppSource(CompiledSource):
    SUFFIX = ".cpp"
    LINE_COMMENT_START = "//"

    @staticmethod
    @ui.hd1("C++", color=ui.BLUE)
    def test(resources: Path, tmp: Path) -> Status:
        file = _copy_test(resources / "test.cpp", tmp)
        cpp = CppSource(file)
        ui.writeln(f"$ {_fmt_cmd(cpp._build_cmd())}")

        bin = cpp.build()
        if isinstance(bin, BuildError):
            ui.writeln(bin.msg, ui.ERROR)
            return Status.fail

        ui.writeln(f"$ {_fmt_cmd(bin.cmd())}")
        return _check_run_status(bin.run())

    def __init__(
        self,
        file: Path,
        extra_files: list[Path] | None = None,
        include: Path | None = None,
        out: Path | None = None,
    ) -> None:
        super().__init__(file)
        self._source = file
        self._extra_files = extra_files or []
        self._include = include
        self._out = out or Path(file.parent, ".build", f"{file.stem}-cpp")
        self._cov_dir = out or Path(file.parent, ".cov", f"{file.stem}-cpp")

    def _ensure_out_dir(self) -> None:
        self._out.parent.mkdir(parents=True, exist_ok=True)

    def _runnable(self) -> Runnable:
        return Binary(self._out)

    def _should_build(self) -> bool:
        return _should_build(self.files, self._out)

    def _build_cmd(self) -> list[str]:
        conf = Config.get().cpp
        cmd = [
            conf.command,
            *conf.get_flags(),
            "-o",
            str(self._out),
        ]
        if self._include:
            cmd.extend(["-I", str(self._include)])
        cmd.extend(str(s) for s in self.files)
        return cmd

    def build_for_coverage(self) -> Runnable | BuildError:
        shutil.rmtree(self._cov_dir, ignore_errors=True)
        self._cov_dir.mkdir(parents=True, exist_ok=True)

        targets = {str(f): str(self._cov_dir / f"{f.stem}.o") for f in self.files}
        for input, obj in targets.items():
            cmd = [Config.get().cpp.command, "-coverage", "-c", "-o", obj, input]
            complete = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if complete.returncode != 0:
                return BuildError(msg=complete.stderr)
        out_bin = self._cov_dir / self._source.stem
        cmd = [
            Config.get().cpp.command,
            "-coverage",
            "-o",
            str(out_bin),
            *targets.values(),
        ]
        complete = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if complete.returncode != 0:
            return BuildError(msg=complete.stderr)
        return Binary(out_bin)

    def compute_coverage(self) -> Coverage | Error:
        complete = subprocess.run(
            [
                "gcov",
                "--json-format",
                "--stdout",
                "-b",
                "-o",
                self._cov_dir,
                self._source,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if complete.returncode != 0 or complete.stderr:
            return Error(msg=complete.stderr)
        try:
            output = json.loads(complete.stdout)
            data = next(
                f for f in output.get("files", []) if f["file"] == str(self._source)
            )
            return compute_gcov_file_coverage(data)
        except Exception as exc:
            return Error(msg=str(exc))

    @property
    def files(self) -> list[Path]:
        return [self._file, *self._extra_files]


@dataclass(kw_only=True, frozen=True)
class Coverage:
    line: float
    branch: float


def compute_gcov_file_coverage(gcov_json: Any) -> Coverage:
    """Compute line and branch coverage from gcov JSON output for a single file."""
    lines = gcov_json.get("lines", [])

    total_lines = executed_lines = 0
    total_branches = executed_branches = 0

    for line in lines:
        # Line coverage
        total_lines += 1
        if line.get("count", 0) > 0 and not line.get("unexecuted_block", False):
            executed_lines += 1

        # Branch coverage
        for branch in line.get("branches", []):
            if branch.get("throw", False):
                continue
            total_branches += 1
            if branch.get("count", 0) > 0:
                executed_branches += 1

    line_cov = (executed_lines / total_lines * 100) if total_lines > 0 else 0.0
    branch_cov = (
        (executed_branches / total_branches * 100) if total_branches > 0 else 0.0
    )

    return Coverage(line=line_cov, branch=branch_cov)


class RustSource(CompiledSource):
    SUFFIX = ".rs"
    LINE_COMMENT_START = "//"

    @staticmethod
    @ui.hd1("Rust", color=ui.BLUE)
    def test(resources: Path, tmp: Path) -> Status:
        file = _copy_test(resources / "test.rs", tmp)
        rs = RustSource(file)
        ui.writeln(f"$ {_fmt_cmd(rs._build_cmd())}")

        bin = rs.build()
        if isinstance(bin, BuildError):
            ui.write(bin.msg, ui.ERROR)
            return Status.fail

        ui.writeln(f"$ {_fmt_cmd(bin.cmd())}")
        return _check_run_status(bin.run())

    def __init__(self, file: Path, out: Path | None = None) -> None:
        super().__init__(file)
        self._out = out or Path(file.parent, ".build", f"{file.stem}-rs")

    def _ensure_out_dir(self) -> None:
        self._out.parent.mkdir(parents=True, exist_ok=True)

    def _runnable(self) -> Runnable:
        return Binary(self._out)

    def _should_build(self) -> bool:
        return _should_build([self._file], self._out)

    def _build_cmd(self) -> list[str]:
        cmd = [
            Config.get().rust.command,
            *Config.get().rust.flags,
            "-o",
            str(self._out),
            str(self._file),
        ]
        return cmd


class JavaSource(CompiledSource):
    SUFFIX = ".java"
    LINE_COMMENT_START = "//"

    @staticmethod
    @ui.hd1("Java", color=ui.BLUE)
    def test(resources: Path, tmp: Path) -> Status:
        file = _copy_test(resources / "test.java", tmp)

        java = JavaSource("Test", file)
        ui.writeln(f"$ {_fmt_cmd(java._build_cmd())}")

        classes = java.build()
        if isinstance(classes, BuildError):
            ui.writeln(classes.msg, ui.ERROR)
            return Status.fail

        ui.writeln(f"$ {_fmt_cmd(classes.cmd())}")
        return _check_run_status(classes.run())

    def __init__(
        self,
        classname: str,
        source: Path,
        outdir: Path | None = None,
    ) -> None:
        super().__init__(source)
        self._classname = classname
        self._source = source
        self._outdir = outdir or Path(source.parent, ".build", f"{source.stem}-java")

    def _ensure_out_dir(self) -> None:
        self._outdir.mkdir(parents=True, exist_ok=True)

    def _runnable(self) -> Runnable:
        return JavaClasses(self._classname, self._outdir)

    def _should_build(self) -> bool:
        return _should_build([self._source], self._outdir)

    def _build_cmd(self) -> list[str]:
        return [Config.get().java.javac, "-d", str(self._outdir), str(self._source)]


class PythonSource(SourceCode):
    SUFFIX = ".py"
    LINE_COMMENT_START = "#"

    @staticmethod
    @ui.hd1("Python", color=ui.BLUE)
    def test(resources: Path, tmp: Path) -> Status:
        file = _copy_test(resources / "test.py", tmp)
        py = PythonSource(file).build()
        ui.writeln(f"$ {_fmt_cmd(py.cmd())}")
        return _check_run_status(py.run())

    def __init__(self, file: Path) -> None:
        super().__init__(file)

    def build(self, *, force: bool = False) -> Python3:
        del force
        return Python3(self._file)


class PDFSource(ABC):
    def __init__(self, source: Path) -> None:
        self._source = source

    @abstractmethod
    def compile(self) -> Path | BuildError: ...

    @ui.work("COMPILE")
    def compile_work(self) -> Result:
        result = self.compile()
        if isinstance(result, Path):
            return Result.success("OK")
        else:
            return Result.fail("FAILED", long_msg=result.msg)

    def iter_lines(self) -> Iterable[str]:
        yield from self._source.open()

    def pdf(self) -> Path | None:
        pdf = self._source.with_suffix(".pdf")
        return pdf if pdf.exists() else None

    def __str__(self) -> str:
        return str(utils.relative_to_cwd(self._source))


class LatexSource(PDFSource):
    @staticmethod
    @ui.hd1("Latex", color=ui.BLUE)
    def test(resources: Path, tmp: Path) -> Status:
        file = _copy_test(resources / "test.tex", tmp)
        latex = LatexSource(file)
        ui.writeln(f"$ {latex._cmd()}")

        if isinstance(r := latex.compile(), BuildError):
            ui.writeln(r.msg, ui.ERROR)
            return Status.fail

        ui.writeln("Success!", ui.GREEN)
        return Status.success

    def __init__(self, source: Path, *, env: dict[str, str] | None = None) -> None:
        super().__init__(source)
        self._env = env

    def _cmd(self) -> str:
        return Template(Config.get().latex.command).substitute(
            TEXNAME=self._source.name,
        )

    def compile(self) -> Path | BuildError:
        complete = subprocess.run(
            self._cmd(),
            cwd=self._source.parent,
            stdin=subprocess.DEVNULL,
            shell=True,
            text=True,
            check=False,
            capture_output=True,
            env=dict(os.environ, **(self._env or {})),
        )
        if complete.returncode != 0:
            msg = complete.stderr
            if msg:
                msg += "\n"
            msg += complete.stdout
            return BuildError(msg=msg)
        return self._source.with_suffix(".pdf")


class TypstSource(PDFSource):
    def __init__(
        self,
        source: Path,
        *,
        sys_inputs: dict[str, str] | None = None,
    ) -> None:
        super().__init__(source)
        self._sys_inputs = sys_inputs or {}

    def compile(self) -> Path | BuildError:
        output = self._source.with_suffix(".pdf")
        fonts = self._source.with_name("fonts")
        try:
            typst.compile(
                self._source,
                output=output,
                sys_inputs=self._sys_inputs,
                font_paths=[fonts],
                ignore_system_fonts=True,
            )
        except Exception as exc:
            return BuildError(msg=str(exc))
        return output


def _fmt_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(s) for s in cmd)


def _check_run_status(r: RunSuccess | RunError) -> Status:
    match r:
        case RunSuccess(_):
            ui.write(r.stdout, ui.GREEN)
            return Status.success
        case RunError(msg, stderr):
            ui.writeln(msg, ui.ERROR)
            ui.writeln(stderr, ui.ERROR)
            return Status.fail


def _copy_test(file: Path, tmp: Path) -> Path:
    ui.writeln(f"$ cp {shlex.quote(str(file))} {shlex.quote(str(tmp))}")
    return Path(shutil.copy2(file, tmp))


def _should_build(sources: list[Path], out: Path) -> bool:
    mtime = max(
        (s.stat().st_mtime for s in sources if s.exists()),
        default=float("inf"),
    )
    if out.is_dir():
        btime = min(
            (s.stat().st_mtime for s in out.iterdir() if s.exists()),
            default=float("-inf"),
        )
    elif out.is_file():
        btime = out.stat().st_mtime
    else:
        btime = float("-inf")
    return btime < mtime
