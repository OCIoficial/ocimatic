from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar, cast

import tomlkit

try:
    __version__ = version("ocimatic")
except PackageNotFoundError:
    __version__ = "not found"

CONTEST_ROOT: Path = Path("/")


@dataclass(kw_only=True)
class CppConfig:
    command: str
    flags: list[str]

    @staticmethod
    def load(conf: dict[Any, Any]) -> CppConfig:
        return CppConfig(command=conf["command"], flags=conf["flags"])


@dataclass(kw_only=True)
class PythonConfig:
    command: str

    @staticmethod
    def load(conf: dict[Any, Any]) -> PythonConfig:
        return PythonConfig(command=conf["command"])


@dataclass(kw_only=True)
class JavaConfig:
    javac: str
    jre: str

    @staticmethod
    def load(conf: dict[Any, Any]) -> JavaConfig:
        return JavaConfig(javac=conf["javac"], jre=conf["jre"])


@dataclass(kw_only=True)
class RustConfig:
    command: str
    flags: list[str]

    @staticmethod
    def load(conf: dict[Any, Any]) -> RustConfig:
        return RustConfig(command=conf["command"], flags=conf["flags"])


@dataclass(kw_only=True)
class LatexConfig:
    command: str

    @staticmethod
    def load(conf: dict[Any, Any]) -> LatexConfig:
        return LatexConfig(command=conf["command"])


class Config:
    HOME_PATH: ClassVar[Path] = Path.home() / ".ocimatic.toml"
    DEFAULT_PATH: ClassVar[Path] = Path(__file__).parent / "resources" / "ocimatic.toml"

    _initialized = False
    cpp: CppConfig
    python: PythonConfig
    java: JavaConfig
    rust: RustConfig
    latex: LatexConfig

    def initialize(self) -> None:
        if self._initialized:
            return

        path = Config.HOME_PATH
        if not path.exists():
            path = Config.DEFAULT_PATH

        with path.open() as f:
            conf = cast(dict[Any, Any], tomlkit.load(f))

        self.cpp = CppConfig.load(conf["cpp"])
        self.python = PythonConfig.load(conf["python"])
        self.java = JavaConfig.load(conf["java"])
        self.rust = RustConfig.load(conf["rust"])
        self.latex = LatexConfig.load(conf["latex"])
        self._initialized = True


CONFIG = Config()
