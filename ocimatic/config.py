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
    detect: bool
    command: str

    @staticmethod
    def load(conf: dict[Any, Any]) -> PythonConfig:
        return PythonConfig(detect=conf["detect"], command=conf["command"])


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


@dataclass(kw_only=True)
class Config:
    HOME_PATH: ClassVar[Path] = Path.home() / ".ocimatic.toml"
    DEFAULT_PATH: ClassVar[Path] = Path(__file__).parent / "resources" / "ocimatic.toml"

    cpp: CppConfig
    python: PythonConfig
    java: JavaConfig
    rust: RustConfig
    latex: LatexConfig

    @staticmethod
    def initialize() -> None:
        if Config._value:
            return

        path = Config.HOME_PATH
        if not path.exists():
            path = Config.DEFAULT_PATH

        with path.open() as f:
            conf = cast(dict[Any, Any], tomlkit.load(f))

        cls._config = Config(
            cpp=CppConfig.load(conf["cpp"]),
            python=PythonConfig.load(conf["python"]),
            java=JavaConfig.load(conf["java"]),
            rust=RustConfig.load(conf["rust"]),
            latex=LatexConfig.load(conf["latex"]),
        )

    @staticmethod
    def get() -> Config:
        assert Config._value, (
            "Configuration not initialized. Call Config.initialize() before Config.get()."
        )
        return Config._value
