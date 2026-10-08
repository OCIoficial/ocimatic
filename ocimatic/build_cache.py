"""Decide when compiled sources must be rebuilt.

Every build writes a stamp recording the command that produced it, the identity of the compiler,
and a hash of every file the build read (declared inputs plus whatever the compiler reports, such
as headers). A build is fresh only if all of these still match, so changes to flags, compilers,
included headers or file contents are all caught, regardless of modification times.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import msgspec

STAMP_VERSION = 1


@dataclass
class BuildError:
    msg: str


class DepsDiscovery(Protocol):
    """A way of finding the dependencies of a build beyond its declared inputs (e.g., headers)."""

    def discover(self) -> list[Path] | BuildError:
        """Return the dependencies. Called after a successful build."""
        ...


@dataclass(frozen=True)
class DepfileDeps:
    """Dependencies listed in a Makefile-style depfile written by the build."""

    path: Path

    def discover(self) -> list[Path] | BuildError:
        try:
            return parse_depfile(self.path.read_text())
        except OSError as e:
            return BuildError(msg=f"Failed to read depfile {self.path}: {e}")


@dataclass(frozen=True)
class CommandDeps:
    """Dependencies listed in a Makefile-style depfile printed to stdout by a command."""

    cmd: list[str]

    def discover(self) -> list[Path] | BuildError:
        if isinstance(out := _run(self.cmd), BuildError):
            return out
        return parse_depfile(out)


@dataclass(frozen=True, kw_only=True)
class BuildRecipe:
    cmd: list[str]
    """Command that performs the build."""

    out: Path
    """File (or directory if `out_is_dir`) produced by the build."""

    stamp: Path
    """Where to record the stamp of the last successful build."""

    inputs: list[Path]
    """Files the build is known to depend on."""

    out_is_dir: bool = False
    """Whether `out` is a directory. It is emptied before every build so no stale outputs remain."""

    deps: DepsDiscovery | None = None
    """How to find dependencies beyond `inputs` after a successful build."""


def ensure_built(recipe: BuildRecipe, *, force: bool = False) -> BuildError | None:
    """Run `recipe` unless its output is up to date (or `force` is set)."""
    if not force and is_fresh(recipe):
        return None

    recipe.stamp.unlink(missing_ok=True)
    recipe.stamp.parent.mkdir(parents=True, exist_ok=True)
    if recipe.out_is_dir:
        shutil.rmtree(recipe.out, ignore_errors=True)
        recipe.out.mkdir(parents=True)
    else:
        recipe.out.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(r := _run(recipe.cmd), BuildError):
        return r

    deps = list(recipe.inputs)
    if recipe.deps is not None:
        if isinstance(discovered := recipe.deps.discover(), BuildError):
            return discovered
        deps.extend(discovered)
    deps = list(dict.fromkeys(p.absolute() for p in deps))

    _write_stamp(recipe.stamp, _current_stamp(recipe, deps))
    return None


def is_fresh(recipe: BuildRecipe) -> bool:
    """Whether the output of `recipe` exists and was built from the current command and files."""
    if not (recipe.out.is_dir() if recipe.out_is_dir else recipe.out.is_file()):
        return False
    stamp = _read_stamp(recipe.stamp)
    if stamp is None:
        return False
    deps = [Path(p) for p in stamp.deps]
    return stamp == _current_stamp(recipe, deps)


def parse_depfile(text: str) -> list[Path]:
    """Return the prerequisites of all rules in a Makefile-style depfile.

    Paths are made absolute relative to the current directory, which is where compilers resolve
    them.
    """
    text = text.replace("\\\r\n", " ").replace("\\\n", " ")
    deps: list[Path] = []
    for line in text.splitlines():
        tokens = _split_depfile_line(line)
        for i, token in enumerate(tokens):
            if token.endswith(":"):
                deps.extend(Path(t).absolute() for t in tokens[i + 1 :])
                break
    return list(dict.fromkeys(deps))


def _split_depfile_line(line: str) -> list[str]:
    r"""Split a depfile line on unescaped whitespace, unescaping `\ `, `\#` and `$$`."""
    tokens: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        if c == "\\" and nxt in (" ", "#"):
            current.append(nxt)
            i += 2
            continue
        if c == "$" and nxt == "$":
            current.append("$")
            i += 2
            continue
        if c == "#" and not current and not tokens:
            break
        if c.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(c)
        i += 1
    if current:
        tokens.append("".join(current))
    return tokens


class _Compiler(msgspec.Struct, frozen=True):
    path: str
    size: int
    mtime_ns: int


class _Stamp(msgspec.Struct, frozen=True):
    version: int
    cmd: list[str]
    compiler: _Compiler | None
    deps: dict[str, str | None]
    """Hash of every dependency, or `None` if it didn't exist."""


def _current_stamp(recipe: BuildRecipe, deps: list[Path]) -> _Stamp:
    return _Stamp(
        version=STAMP_VERSION,
        cmd=recipe.cmd,
        compiler=_compiler_identity(recipe.cmd[0]),
        deps={str(p): _hash_file(p) for p in deps},
    )


def _compiler_identity(program: str) -> _Compiler | None:
    path = shutil.which(program)
    if path is None:
        return None
    try:
        st = Path(path).resolve().stat()
    except OSError:
        return None
    return _Compiler(path=path, size=st.st_size, mtime_ns=st.st_mtime_ns)


def _hash_file(path: Path) -> str | None:
    try:
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return None


def _read_stamp(path: Path) -> _Stamp | None:
    try:
        return msgspec.json.decode(path.read_bytes(), type=_Stamp)
    except (OSError, msgspec.DecodeError):
        return None


def _write_stamp(path: Path, stamp: _Stamp) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(msgspec.json.encode(stamp))
    tmp.replace(path)


def _run(cmd: list[str]) -> str | BuildError:
    """Run `cmd`, returning its stdout, or its stderr as an error if it fails."""
    try:
        complete = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return BuildError(msg=str(e))
    if complete.returncode != 0:
        return BuildError(msg=complete.stderr)
    return complete.stdout
