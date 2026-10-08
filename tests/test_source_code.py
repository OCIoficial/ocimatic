from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from ocimatic.config import Config
from ocimatic.env import Env
from ocimatic.runnable import Runnable, RunSuccess
from ocimatic.source_code import (
    BuildError,
    CppSource,
    JavaSource,
    RustSource,
    SourceCode,
)

from .conftest import UseEnv
from .text import block
from .tree import write_file, write_tree


class Compilations:
    """Counts the compiler invocations that build something (as opposed to listing dependencies)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, compiler: str) -> None:
        self.count = 0
        real_run = subprocess.run

        def run(cmd: list[str], *args: object, **kwargs: object) -> object:
            if cmd[0] == compiler and "-MM" not in cmd:
                self.count += 1
            return real_run(cmd, *args, **kwargs)  # type: ignore[call-overload]

        monkeypatch.setattr(subprocess, "run", run)


def _require(program: str) -> None:
    """Skip the test if `program` isn't installed, or fail it if `OCIMATIC_REQUIRE_TOOLCHAINS` is set.

    CI sets the variable so a missing compiler can't silently skip these tests.
    """
    if shutil.which(program) is not None:
        return
    if os.environ.get("OCIMATIC_REQUIRE_TOOLCHAINS"):
        pytest.fail(
            f"{program} is not installed (required by OCIMATIC_REQUIRE_TOOLCHAINS)",
        )
    pytest.skip(f"{program} is not installed")


def _build_and_run(source: SourceCode) -> str:
    runnable = source.build()
    assert not isinstance(runnable, BuildError), runnable.msg
    assert isinstance(runnable, Runnable)
    result = runnable.run()
    assert isinstance(result, RunSuccess)
    return result.stdout


@pytest.fixture
def env(tmp_path: Path, use_env: UseEnv) -> Iterator[Env]:
    with use_env(cwd=tmp_path) as env:
        yield env


@pytest.fixture
def cpp(monkeypatch: pytest.MonkeyPatch) -> Compilations:
    command = Config().cpp.command
    _require(command)
    monkeypatch.delenv("OCIMATIC_CPP_FLAGS", raising=False)
    return Compilations(monkeypatch, command)


def test_cpp_rebuilds_when_manager_header_changes(
    tmp_path: Path,
    env: Env,
    cpp: Compilations,
) -> None:
    write_tree(
        tmp_path,
        {
            "managers": {
                "task.h": block(
                    """
                    int answer();
                    #define GREETING 1
                    """,
                ),
                "grader.cpp": block(
                    """
                    #include <cstdio>
                    #include "task.h"
                    int main() { printf("%d %d", GREETING, answer()); }
                    """,
                ),
            },
            "solutions": {
                "sol.cpp": block(
                    """
                    #include "task.h"
                    int answer() { return 42; }
                    """,
                ),
            },
        },
    )
    managers = tmp_path / "managers"
    source = CppSource(
        tmp_path / "solutions" / "sol.cpp",
        extra_files=[managers / "grader.cpp"],
        include=managers,
    )

    assert _build_and_run(source) == "1 42"
    assert _build_and_run(source) == "1 42"
    assert cpp.count == 1

    write_file(
        managers / "task.h",
        block(
            """
            int answer();
            #define GREETING 2
            """,
        ),
    )
    assert _build_and_run(source) == "2 42"
    assert cpp.count == 2


def test_cpp_rebuilds_when_local_header_changes(
    tmp_path: Path,
    env: Env,
    cpp: Compilations,
) -> None:
    write_tree(
        tmp_path,
        {
            "gen.cpp": block(
                """
                #include <cstdio>
                #include "lib.h"
                int main() { printf("%d", N); }
                """,
            ),
            "lib.h": "#define N 1\n",
        },
    )
    source = CppSource(tmp_path / "gen.cpp")

    assert _build_and_run(source) == "1"
    write_file(tmp_path / "lib.h", "#define N 2\n")
    assert _build_and_run(source) == "2"
    assert cpp.count == 2


def test_cpp_rebuilds_when_flags_change(
    tmp_path: Path,
    env: Env,
    cpp: Compilations,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_tree(
        tmp_path,
        {
            "sol.cpp": block(
                """
                #include <cstdio>
                int main() { printf("%d", N); }
                """,
            ),
        },
    )
    source = CppSource(tmp_path / "sol.cpp")

    monkeypatch.setenv("OCIMATIC_CPP_FLAGS", "-O2 -DN=1")
    assert _build_and_run(source) == "1"
    assert _build_and_run(source) == "1"
    monkeypatch.setenv("OCIMATIC_CPP_FLAGS", "-O2 -DN=2")
    assert _build_and_run(source) == "2"
    assert cpp.count == 2


def test_cpp_build_error(tmp_path: Path, env: Env, cpp: Compilations) -> None:
    write_tree(tmp_path, {"sol.cpp": "int main() { return x; }\n"})

    assert isinstance(CppSource(tmp_path / "sol.cpp").build(), BuildError)


def test_rust_rebuilds_when_module_changes(
    tmp_path: Path,
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = Config().rust.command
    _require(command)
    rustc = Compilations(monkeypatch, command)
    write_tree(
        tmp_path,
        {
            "sol.rs": block(
                """
                mod util;
                fn main() { print!("{}", util::N); }
                """,
            ),
            "util.rs": "pub const N: i32 = 1;\n",
        },
    )
    source = RustSource(tmp_path / "sol.rs")

    assert _build_and_run(source) == "1"
    assert _build_and_run(source) == "1"
    write_file(tmp_path / "util.rs", "pub const N: i32 = 2;\n")
    assert _build_and_run(source) == "2"
    assert rustc.count == 2


def test_java_removes_stale_classes(tmp_path: Path, env: Env) -> None:
    _require(Config().java.javac)
    _require(Config().java.jre)
    write_tree(
        tmp_path,
        {
            "sol.java": block(
                """
                public class sol {
                    public static void main(String[] a) { System.out.print(1); }
                }
                class Helper {}
                """,
            ),
        },
    )
    source = JavaSource("sol", tmp_path / "sol.java")

    assert _build_and_run(source) == "1"
    classes = tmp_path / ".build" / "sol-java"
    assert (classes / "Helper.class").exists()

    write_file(
        tmp_path / "sol.java",
        block(
            """
            public class sol {
                public static void main(String[] a) { System.out.print(2); }
            }
            """,
        ),
    )
    assert _build_and_run(source) == "2"
    assert not (classes / "Helper.class").exists()
