"""Ocimatic is a tool for automating tasks related to the creation of problems for the Chilean Olympiad in Informatics (OCI).

:license: Beer-Ware, see LICENSE.rst for more details.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import tomlkit

contest_root: Path = Path("/")


@dataclass(kw_only=True)
class CppConfig:
    command: str
    flags: list[str]

    @staticmethod
    def load(config: dict[Any, Any]) -> CppConfig:
        return CppConfig(command=config["command"], flags=config["flags"])


@dataclass(kw_only=True)
class PythonConfig:
    command: str

    @staticmethod
    def load(config: dict[Any, Any]) -> PythonConfig:
        return PythonConfig(command=config["command"])


@dataclass(kw_only=True)
class JavaConfig:
    javac: str
    jre: str

    @staticmethod
    def load(config: dict[Any, Any]) -> JavaConfig:
        return JavaConfig(javac=config["javac"], jre=config["jre"])


@dataclass(kw_only=True)
class RustConfig:
    command: str
    flags: list[str]

    @staticmethod
    def load(config: dict[Any, Any]) -> RustConfig:
        return RustConfig(command=config["command"], flags=config["flags"])


@dataclass(kw_only=True)
class LatexConfig:
    command: str
    flags: list[str]

    @staticmethod
    def load(config: dict[Any, Any]) -> LatexConfig:
        return LatexConfig(command=config["command"], flags=config["flags"])


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
            config = cast(dict[Any, Any], tomlkit.load(f))

        self.cpp = CppConfig.load(config["cpp"])
        self.python = PythonConfig.load(config["python"])
        self.java = JavaConfig.load(config["java"])
        self.rust = RustConfig.load(config["rust"])
        self.latex = LatexConfig.load(config["latex"])
        self.initialized = True


config = Config()
