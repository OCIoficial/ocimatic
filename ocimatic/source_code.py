from __future__ import annotations

import re
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import ocimatic
from ocimatic import ui
from ocimatic.runnable import Binary, JavaClasses, Python3, Runnable


@dataclass
class BuildError:
    msg: str


@dataclass(frozen=True, kw_only=True, slots=True)
class ShouldFail:
    REGEX = re.compile(r"\s+should-fail\s*=\s*\[(\s*st\d+\s*(,\s*st\d+\s*)*(,\s*)?)\]")
    subtasks: set[int]

    @staticmethod
    def parse(comment: str) -> ShouldFail | None:
        m = ShouldFail.REGEX.match(comment)
        if not m:
            return None
        subtasks = {int(st.strip().removeprefix("st")) for st in m.group(1).split(",")}
        return ShouldFail(subtasks=subtasks)


@dataclass(frozen=True, kw_only=True, slots=True)
class ShouldPass:
    REGEX = re.compile(r"\s+should-pass\s*=\s*\[(\s*st\d+\s*(,\s*st\d+\s*)*(,\s*)?)\]")
    subtasks: set[int]

    @staticmethod
    def parse(comment: str) -> ShouldPass | None:
        m = ShouldPass.REGEX.match(comment)
        if not m:
            return None
        subtasks = {int(st.strip().removeprefix("st")) for st in m.group(1).split(",")}
        return ShouldPass(subtasks=subtasks)


OcimaticComment = ShouldFail | ShouldPass


class SourceCode(ABC):
    LINE_COMMENT_START: str

    def __init__(self, file: Path) -> None:
        relative_path = relative_to_contest_root(file)
        self._file = file
        self.name = str(relative_path)
        self.comments = list(parse_comments(file, self.__class__.LINE_COMMENT_START))

    def __str__(self) -> str:
        return self.name

    @property
    def file(self) -> Path:
        return self._file

    @staticmethod
    def should_build(sources: list[Path], out: Path) -> bool:
        mtime = max(
            (s.stat().st_mtime for s in sources if s.exists()),
            default=float("inf"),
        )
        btime = out.stat().st_mtime if out.exists() else float("-inf")
        return btime < mtime

    @abstractmethod
    def build(self, *, force: bool = False) -> Runnable | BuildError:
        ...


class CppSource(SourceCode):
    LINE_COMMENT_START = "//"

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

    def build_cmd(self) -> list[str]:
        cmd = ["g++", "-std=c++17", "-O2", "-o", str(self._out)]
        if self._include:
            cmd.extend(["-I", str(self._include)])
        cmd.extend(str(s) for s in self.files)
        return cmd

    def build(self, *, force: bool = False) -> Binary | BuildError:
        self._out.parent.mkdir(parents=True, exist_ok=True)
        if force or CppSource.should_build(self.files, self._out):
            cmd = self.build_cmd()
            complete = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if complete.returncode != 0:
                return BuildError(msg=complete.stderr)
        return Binary(self._out)

    @property
    def files(self) -> list[Path]:
        return [self._file, *self._extra_files]


class RustSource(SourceCode):
    LINE_COMMENT_START = "//"

    def __init__(self, file: Path, out: Path | None = None) -> None:
        super().__init__(file)
        self._out = out or Path(file.parent, ".build", f"{file.stem}-rs")

    def build_cmd(self) -> list[str]:
        cmd = ["rustc", "--edition=2021", "-O", "-o", str(self._out), str(self._file)]
        return cmd

    def build(self, *, force: bool = False) -> Binary | BuildError:
        self._out.parent.mkdir(parents=True, exist_ok=True)
        if force or RustSource.should_build([self._file], self._out):
            cmd = self.build_cmd()
            complete = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if complete.returncode != 0:
                return BuildError(msg=complete.stderr)
        return Binary(self._out)


class JavaSource(SourceCode):
    LINE_COMMENT_START = "//"

    def __init__(self, classname: str, source: Path, out: Path | None = None) -> None:
        super().__init__(source)
        self._classname = classname
        self._source = source
        self._out = out or Path(source.parent, ".build", f"{source.stem}-java")

    def build_cmd(self) -> list[str]:
        return ["javac", "-d", str(self._out), str(self._source)]

    def build(self, *, force: bool = False) -> JavaClasses | BuildError:
        if force or JavaSource.should_build([self._source], self._out):
            self._out.mkdir(parents=True, exist_ok=True)
            cmd = self.build_cmd()
            complete = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if complete.returncode != 0:
                return BuildError(msg=complete.stderr)
        return JavaClasses(self._classname, self._out)


class PythonSource(SourceCode):
    LINE_COMMENT_START = "#"

    def __init__(self, file: Path) -> None:
        super().__init__(file)

    def build(self, *, force: bool = False) -> Python3:
        del force
        return Python3(self._file)


def parse_comments(file: Path, comment_start: str) -> Iterator[OcimaticComment]:
    for m in comment_iter(file, comment_start):
        for parser in [ShouldFail.parse, ShouldPass.parse]:
            parsed = parser(m.group(1))
            if parsed:
                yield parsed
                break
        else:
            path = relative_to_contest_root(file)
            ui.fatal_error(f"Invalid comment `{m.group(0)}` in {path}")


def comment_iter(file_path: Path, comment_start: str) -> Iterator[re.Match[str]]:
    pattern = re.compile(rf"\s*{comment_start}\s*@ocimatic(.*)")
    with file_path.open() as file:
        for line in file:
            m = pattern.match(line)
            if m:
                yield m


class LatexSource:
    def __init__(self, source: Path) -> None:
        self._source = source

    def iter_lines(self) -> Iterable[str]:
        yield from self._source.open()

    def compile(self) -> Path | BuildError:
        name = self._source.name
        parent = self._source.parent
        cmd = f"pdflatex --shell-escape -interaciton=batchmode {name}"
        complete = subprocess.run(
            cmd,
            cwd=parent,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if complete.returncode != 0:
            return BuildError(msg=complete.stderr)
        return self._source.with_suffix(".pdf")

    @property
    def pdf(self) -> Path | None:
        pdf = self._source.with_suffix(".pdf")
        return pdf if pdf.exists() else None

    def __str__(self) -> str:
        return str(relative_to_contest_root(self._source))


def relative_to_contest_root(path: Path) -> Path:
    try:
        return path.relative_to(ocimatic.config["contest_root"])
    except ValueError:
        return path
