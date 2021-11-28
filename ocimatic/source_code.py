import subprocess
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional, Union

from ocimatic.runnable import Binary, JavaClasses, Python3, Runnable


@dataclass
class BuildError:
    msg: str


class SourceCode(ABC):
    def __init__(self, source: Path, extra_sources: List[Path]):
        self._source = source
        self._extra_sources = extra_sources

    @property
    def sources(self) -> List[Path]:
        sources = [self._source]
        sources.extend(self._extra_sources)
        return sources

    @property
    def name(self) -> str:
        return str(self._source)

    def __str__(self) -> str:
        return str(self._source)

    def mtime(self) -> float:
        return max(s.stat().st_mtime for s in self.sources)

    @abstractmethod
    def build(self, force: bool = False) -> Union[Runnable, BuildError]:
        raise NotImplementedError("Class %s doesn't implement build()" % (self.__class__.__name__))


class CppSource(SourceCode):
    def __init__(self,
                 source: Path,
                 extra_sources: List[Path] = [],
                 include: Path = None,
                 out: Path = None):
        super().__init__(source, extra_sources)
        self._include = include
        self._out = out or Path(source.parent, '.build', source.stem)

    def build_time(self) -> float:
        if self._out.exists():
            return self._out.stat().st_mtime
        else:
            return float("-inf")

    def build_cmd(self) -> List[str]:
        cmd = ['g++', '-std=c++11', '-O2', '-o', str(self._out)]
        if self._include:
            cmd.extend(['-I', str(self._include)])
        cmd.extend(str(s) for s in self.sources)
        return cmd

    def build(self, force: bool = False) -> Union[Binary, BuildError]:
        self._out.parent.mkdir(parents=True, exist_ok=True)
        if force or self.build_time() < self.mtime():
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
        super().__init__(source, [])
        self._classname = classname
        self._out = out or Path(source.parent, '.build', f"{source.stem}.classes")

    def build_time(self) -> float:
        if self._out.exists():
            return self._out.stat().st_mtime
        else:
            return float("-inf")

    def build_cmd(self) -> List[str]:
        cmd = ['javac', '-d', str(self._out)]
        cmd.extend(str(s) for s in self.sources)
        return cmd

    def build(self, force: bool = False) -> Union[JavaClasses, BuildError]:
        if force or self.build_time() < self.mtime():
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
        super().__init__(source, [])

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
