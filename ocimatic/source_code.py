import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set

import ocimatic
from ocimatic.runnable import Binary, JavaClasses, Python3, Runnable


@dataclass
class BuildError:
    msg: str


@dataclass(frozen=True, kw_only=True, slots=True)
class SolutionComment:
    should_fail: Set[int] | None


class SourceCode(ABC):

    def __init__(self, file: Path):
        self._file = file
        self.name = str(file.relative_to(ocimatic.config['contest_root']))
        self.comment = extract_comment(file, SourceCode.line_comment_str())

    def __str__(self) -> str:
        return self.name

    @property
    def file(self) -> Path:
        return self._file

    @staticmethod
    def should_build(sources: List[Path], out: Path) -> bool:
        mtime = max((s.stat().st_mtime for s in sources if s.exists()), default=float("inf"))
        btime = out.stat().st_mtime if out.exists() else float("-inf")
        return btime < mtime

    @abstractmethod
    def build(self, force: bool = False) -> Runnable | BuildError:
        ...

    @classmethod
    @abstractmethod
    def line_comment_str(cls) -> str:
        ...


class CppSource(SourceCode):

    def __init__(self,
                 file: Path,
                 extra_files: List[Path] = [],
                 include: Optional[Path] = None,
                 out: Optional[Path] = None):
        super().__init__(file)
        self._source = file
        self._extra_files = extra_files
        self._include = include
        self._out = out or Path(file.parent, '.build', f'{file.stem}-cpp')

    def build_cmd(self) -> List[str]:
        cmd = ['g++', '-std=c++17', '-O2', '-o', str(self._out)]
        if self._include:
            cmd.extend(['-I', str(self._include)])
        cmd.extend(str(s) for s in self.files)
        return cmd

    def build(self, force: bool = False) -> Binary | BuildError:
        self._out.parent.mkdir(parents=True, exist_ok=True)
        if force or CppSource.should_build(self.files, self._out):
            cmd = self.build_cmd()
            complete = subprocess.run(cmd,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE,
                                      text=True,
                                      check=False)
            if complete.returncode != 0:
                return BuildError(msg=complete.stderr)
        return Binary(self._out)

    @property
    def files(self) -> List[Path]:
        return [self._file] + self._extra_files

    @classmethod
    def line_comment_str(cls) -> str:
        return "//"


class RustSource(SourceCode):

    def __init__(self, file: Path, out: Optional[Path] = None):
        super().__init__(file)
        self._out = out or Path(file.parent, '.build', f'{file.stem}-rs')

    def build_cmd(self) -> List[str]:
        cmd = ['rustc', '--edition=2021', '-O', '-o', str(self._out), str(self._file)]
        return cmd

    def build(self, force: bool = False) -> Binary | BuildError:
        self._out.parent.mkdir(parents=True, exist_ok=True)
        if force or RustSource.should_build([self._file], self._out):
            cmd = self.build_cmd()
            complete = subprocess.run(cmd,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE,
                                      text=True,
                                      check=False)
            if complete.returncode != 0:
                return BuildError(msg=complete.stderr)
        return Binary(self._out)

    @classmethod
    def line_comment_str(cls) -> str:
        return "//"


class JavaSource(SourceCode):

    def __init__(self, classname: str, source: Path, out: Optional[Path] = None):
        super().__init__(source)
        self._classname = classname
        self._source = source
        self._out = out or Path(source.parent, '.build', f"{source.stem}-java")

    def build_cmd(self) -> List[str]:
        return ['javac', '-d', str(self._out), str(self._source)]

    def build(self, force: bool = False) -> JavaClasses | BuildError:
        if force or JavaSource.should_build([self._source], self._out):
            self._out.mkdir(parents=True, exist_ok=True)
            cmd = self.build_cmd()
            complete = subprocess.run(cmd,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE,
                                      text=True,
                                      check=False)
            if complete.returncode != 0:
                return BuildError(msg=complete.stderr)
        return JavaClasses(self._classname, self._out)

    @classmethod
    def line_comment_str(cls) -> str:
        return "//"


class PythonSource(SourceCode):

    def __init__(self, file: Path):
        super().__init__(file)

    def build(self, force: bool = False) -> Python3:
        del force
        return Python3(self._file)

    @classmethod
    def line_comment_str(cls) -> str:
        return "#"


def extract_comment(file: Path, comment_str: str) -> SolutionComment:
    first_line = next(open(file))
    if not first_line:
        return SolutionComment(should_fail=None)

    pattern = rf"\s*{comment_str}\s*@ocimatic\s+should-fail\s*=\s*\[((\s*st\d+)(,\s*st\d+)(\s*,)?\s*)\]"
    m = re.match(pattern, first_line.strip())

    if not m:
        return SolutionComment(should_fail=None)

    should_fail = set(int(st.strip().removeprefix('st')) for st in m.group(1).split(","))

    return SolutionComment(should_fail=should_fail)


class LatexSource:

    def __init__(self, source: Path):
        self._source = source

    def iter_lines(self) -> Iterable[str]:
        yield from self._source.open()

    def compile(self) -> Optional[Path]:
        name = self._source.name
        parent = self._source.parent
        cmd = f"cd {parent} && pdflatex --shell-escape -interaciton=batchmode {name}"
        complete = subprocess.run(cmd,
                                  shell=True,
                                  stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  check=False)
        if complete.returncode != 0:
            return None
        return self._source.with_suffix('.pdf')

    @property
    def pdf(self) -> Optional[Path]:
        pdf = self._source.with_suffix('.pdf')
        return pdf if pdf.exists() else None

    def __str__(self) -> str:
        return str(self._source.relative_to(ocimatic.config['contest_root']))
