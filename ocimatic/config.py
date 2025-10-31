from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import ClassVar

import msgspec

from ocimatic import ui

try:
    __version__ = version("ocimatic")
except PackageNotFoundError:
    __version__ = "not found"

CONTEST_ROOT: Path = Path("/")


class CppConfig(msgspec.Struct, kw_only=True, frozen=True):
    command: str
    flags: list[str]
    sanitize_flags: list[str]


class PythonConfig(msgspec.Struct, kw_only=True, frozen=True):
    detect: bool
    command: str


class JavaConfig(msgspec.Struct, kw_only=True, frozen=True):
    javac: str
    jre: str


class RustConfig(msgspec.Struct, kw_only=True, frozen=True):
    command: str
    flags: list[str]


class LatexConfig(msgspec.Struct, kw_only=True, frozen=True):
    command: str


class Config(msgspec.Struct, kw_only=True, frozen=True):
    _value: ClassVar[Config | None] = None
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

        try:
            Config._value = msgspec.toml.decode(path.read_text(), type=Config)
        except Exception as e:
            ui.fatal_error(
                f"Failed to load configuration from {path}: {e}\n"
                "You can regenerate the default configuration with `ocimatic setup`.",
            )

    @staticmethod
    def get() -> Config:
        assert Config._value, (
            "Configuration not initialized. Call Config.initialize() before Config.get()."
        )
        return Config._value
