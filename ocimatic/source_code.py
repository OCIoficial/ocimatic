import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

from ocimatic.runnable import Binary, JavaClasses, Python3, Runnable


@dataclass
class BuildError:
    msg: str


class SourceCode(ABC):
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def should_build(sources: List[Path], out: Path) -> bool:
        mtime = max(s.stat().st_mtime for s in sources)
        btime = out.stat().st_mtime if out.exists() else float("-inf")
        return btime < mtime

    @abstractmethod
    def build(self, force: bool = False) -> Union[Runnable, BuildError]:
        raise NotImplementedError("Class %s doesn't implement build()" % (self.__class__.__name__))


class CppSource(SourceCode):
    def __init__(self,
                 source: Path,
                 extra_sources: List[Path] = [],
                 include: Path = None,
                 out: Path = None):
        super().__init__(str(source))
        self._sources = [source] + extra_sources
        self._include = include
        self._out = out or Path(source.parent, '.build', source.stem)

    def build_cmd(self) -> List[str]:
        cmd = ['g++', '-std=c++11', '-O2', '-o', str(self._out)]
        if self._include:
            cmd.extend(['-I', str(self._include)])
        cmd.extend(str(s) for s in self._sources)
        return cmd

    def build(self, force: bool = False) -> Union[Binary, BuildError]:
        self._out.parent.mkdir(parents=True, exist_ok=True)
        if force or CppSource.should_build(self._sources, self._out):
            cmd = self.build_cmd()
            complete = subprocess.run(cmd,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE,
                                      text=True,
                                      check=False)
            if complete.returncode != 0:
                return BuildError(msg=complete.stderr)
        return Binary(self._out)


class JavaSource(SourceCode):
    def __init__(self, classname: str, source: Path, out: Path = None):
        super().__init__(str(source))
        self._classname = classname
        self._source = source
        self._out = out or Path(source.parent, '.build', f"{source.stem}.classes")

    def build_cmd(self) -> List[str]:
        return ['javac', '-d', str(self._out), str(self._source)]

    def build(self, force: bool = False) -> Union[JavaClasses, BuildError]:
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


class PythonSource(SourceCode):
    def __init__(self, source: Path):
        super().__init__(str(source))
        self._source = source

    def build(self, _force: bool = False) -> Python3:
        return Python3(self._source)


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
        return str(self._source)
