from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar
from collections.abc import Callable

import msgspec
import tomlkit

from ocimatic import ui

try:
    __version__ = version("ocimatic")
except PackageNotFoundError:
    __version__ = "not found"

CONTEST_ROOT: Path = Path("/")


def field[T](c: Callable[[], T]) -> T:
    return msgspec.field(default_factory=c)


class CppConfig(msgspec.Struct, kw_only=True, frozen=True):
    command: str = "clang++"
    flags: list[str] = field(lambda: ["-O2", "-std=c++20"])


class PythonConfig(msgspec.Struct, kw_only=True, frozen=True):
    detect: bool = True
    command: str = "python3"


class JavaConfig(msgspec.Struct, kw_only=True, frozen=True):
    javac: str = "javac"
    jre: str = "java"


class RustConfig(msgspec.Struct, kw_only=True, frozen=True):
    command: str = "rustc"
    flags: list[str] = field(lambda: ["--edition=2024", "-O"])


class LatexConfig(msgspec.Struct, kw_only=True, frozen=True):
    command: str = 'texfot pdflatex -file-line-error -shell-escape "$TEXNAME"'


class Config(msgspec.Struct, kw_only=True, frozen=True):
    _value: ClassVar[Config | None] = None
    HOME_PATH: ClassVar[Path] = Path.home() / ".ocimatic.toml"
    TEMPLATE_PATH: ClassVar[Path] = (
        Path(__file__).parent / "resources" / "ocimatic.toml"
    )

    cpp: CppConfig = CppConfig()
    python: PythonConfig = PythonConfig()
    java: JavaConfig = JavaConfig()
    rust: RustConfig = RustConfig()
    latex: LatexConfig = LatexConfig()

    @staticmethod
    def default_toml_document() -> tomlkit.TOMLDocument:
        # We keep the default values in the definition of the structs,
        # but the template contains comments, so we merge defaults values
        # into the template.
        doc = tomlkit.load(Config.TEMPLATE_PATH.open("r"))
        _merge_toml(doc, msgspec.to_builtins(Config()))
        return doc

    @staticmethod
    def initialize() -> None:
        if Config._value:
            return

        path = Config.HOME_PATH
        if Config.HOME_PATH.exists():
            try:
                Config._value = msgspec.toml.decode(path.read_text(), type=Config)
            except Exception as e:
                ui.fatal_error(
                    f"Failed to load configuration from {path}: {e}\n"
                    "You can regenerate the default configuration with `ocimatic setup`.",
                )
        else:
            Config._value = Config()

    @staticmethod
    def get() -> Config:
        assert Config._value, (
            "Configuration not initialized. Call Config.initialize() before Config.get()."
        )
        return Config._value


def _merge_toml(doc: Any, data: Any) -> None:
    """Merge data into a `tomlkit.TOMLDocument`.

    The data is expected to have the same shape as the toml
    """
    for key, value in data.items():
        if key in doc and isinstance(doc[key], dict) and isinstance(value, dict):
            _merge_toml(doc[key], value)
        else:
            doc[key] = value
