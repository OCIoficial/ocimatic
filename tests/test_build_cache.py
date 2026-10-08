from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ocimatic.build_cache import (
    BuildError,
    BuildRecipe,
    CommandDeps,
    DepfileDeps,
    DepsDiscovery,
    ensure_built,
    parse_depfile,
)

from .text import block
from .tree import write_tree

# A stand-in for a compiler. `cc.py [--depfile D] [--deps] [-X...] OUT INPUT...` concatenates the inputs
# into OUT (failing if one contains `ERROR`), and reports every `#include "x"` as a dependency,
# either in the depfile D or, with `--deps`, as a depfile on stdout without building anything.
# Flags starting with `-X` are ignored.
FAKE_CC = block(
    """
    import sys
    from pathlib import Path

    args = sys.argv[1:]
    depfile = None
    if args[0] == "--depfile":
        depfile, args = Path(args[1]), args[2:]
    args = [a for a in args if not a.startswith("-X")]
    deps_only = args[0] == "--deps"
    if deps_only:
        args = args[1:]
    out, inputs = Path(args[0]), [Path(a) for a in args[1:]]
    deps = []
    for i in inputs:
        for line in i.read_text().splitlines():
            if line.startswith('#include "'):
                deps.append(str(i.parent / line.split('"')[1]).replace(" ", "\\\\ "))
    rule = f"{out}: " + " \\\\\\n  ".join(deps) + "\\n"
    if deps_only:
        print(rule)
        sys.exit(0)
    content = "".join(i.read_text() for i in inputs)
    if "ERROR" in content:
        print("compile error", file=sys.stderr)
        sys.exit(1)
    out.write_text(content)
    if depfile:
        depfile.write_text(rule)
    """,
)


class Builds:
    """Counts the builds performed by the fake compiler."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        real_run = subprocess.run

        def run(cmd: list[str], *args: object, **kwargs: object) -> object:
            if "--deps" not in cmd:
                self.count += 1
            return real_run(cmd, *args, **kwargs)  # type: ignore[call-overload]

        monkeypatch.setattr(subprocess, "run", run)


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> Builds:
    return Builds(monkeypatch)


@pytest.fixture
def cc(tmp_path: Path) -> Path:
    write_tree(tmp_path, {"cc.py": FAKE_CC})
    return tmp_path / "cc.py"


def _recipe(
    cc: Path,
    tmp_path: Path,
    *inputs: Path,
    flags: tuple[str, ...] = (),
    deps: str = "depfile",
) -> BuildRecipe:
    out = tmp_path / ".build" / "out"
    depfile = tmp_path / ".build" / "out.d"
    files = [str(i) for i in inputs]
    cmd = [sys.executable, str(cc), *flags, str(out), *files]
    discovery: DepsDiscovery
    if deps == "depfile":
        cmd = [*cmd[:2], "--depfile", str(depfile), *cmd[2:]]
        discovery = DepfileDeps(depfile)
    else:
        discovery = CommandDeps([sys.executable, str(cc), "--deps", str(out), *files])
    return BuildRecipe(
        cmd=cmd,
        out=out,
        stamp=tmp_path / ".build" / "out.stamp.json",
        inputs=list(inputs),
        deps=discovery,
    )


def test_builds_once(cc: Path, tmp_path: Path, builds: Builds) -> None:
    write_tree(tmp_path, {"a.c": "a\n"})
    src = tmp_path / "a.c"
    recipe = _recipe(cc, tmp_path, src)

    assert ensure_built(recipe) is None
    assert ensure_built(recipe) is None
    assert builds.count == 1
    assert recipe.out.read_text() == "a\n"


def test_force_rebuilds(cc: Path, tmp_path: Path, builds: Builds) -> None:
    write_tree(tmp_path, {"a.c": "a\n"})
    src = tmp_path / "a.c"
    recipe = _recipe(cc, tmp_path, src)

    ensure_built(recipe)
    ensure_built(recipe, force=True)
    assert builds.count == 2


def test_rebuilds_on_content_change_even_if_mtime_is_older(
    cc: Path,
    tmp_path: Path,
    builds: Builds,
) -> None:
    write_tree(tmp_path, {"a.c": "a\n"})
    src = tmp_path / "a.c"
    recipe = _recipe(cc, tmp_path, src)
    ensure_built(recipe)

    src.write_text("b\n")
    os.utime(src, (0, 0))
    ensure_built(recipe)
    assert builds.count == 2
    assert recipe.out.read_text() == "b\n"


def test_touch_without_change_does_not_rebuild(
    cc: Path,
    tmp_path: Path,
    builds: Builds,
) -> None:
    write_tree(tmp_path, {"a.c": "a\n"})
    src = tmp_path / "a.c"
    recipe = _recipe(cc, tmp_path, src)
    ensure_built(recipe)

    src.touch()
    ensure_built(recipe)
    assert builds.count == 1


def test_rebuilds_on_command_change(cc: Path, tmp_path: Path, builds: Builds) -> None:
    write_tree(tmp_path, {"a.c": "a\n"})
    src = tmp_path / "a.c"
    ensure_built(_recipe(cc, tmp_path, src))
    ensure_built(_recipe(cc, tmp_path, src, flags=("-Xfoo",)))
    ensure_built(_recipe(cc, tmp_path, src, flags=("-Xfoo",)))
    assert builds.count == 2


@pytest.mark.parametrize("deps", ["depfile", "command"])
def test_rebuilds_on_header_change(
    cc: Path,
    tmp_path: Path,
    builds: Builds,
    deps: str,
) -> None:
    write_tree(
        tmp_path,
        {"inc dir": {"h.h": "h1\n"}, "a.c": '#include "inc dir/h.h"\n'},
    )
    header = tmp_path / "inc dir" / "h.h"
    recipe = _recipe(cc, tmp_path, tmp_path / "a.c", deps=deps)

    ensure_built(recipe)
    ensure_built(recipe)
    assert builds.count == 1

    header.write_text("h2\n")
    ensure_built(recipe)
    assert builds.count == 2


def test_rebuilds_when_output_is_missing(
    cc: Path,
    tmp_path: Path,
    builds: Builds,
) -> None:
    write_tree(tmp_path, {"a.c": "a\n"})
    src = tmp_path / "a.c"
    recipe = _recipe(cc, tmp_path, src)
    ensure_built(recipe)

    recipe.out.unlink()
    ensure_built(recipe)
    assert builds.count == 2


def test_rebuilds_on_corrupt_stamp(cc: Path, tmp_path: Path, builds: Builds) -> None:
    write_tree(tmp_path, {"a.c": "a\n"})
    src = tmp_path / "a.c"
    recipe = _recipe(cc, tmp_path, src)
    ensure_built(recipe)

    recipe.stamp.write_text("{not json")
    ensure_built(recipe)
    assert builds.count == 2


def test_failed_build_leaves_no_stamp(cc: Path, tmp_path: Path, builds: Builds) -> None:
    write_tree(tmp_path, {"a.c": "a\n"})
    src = tmp_path / "a.c"
    recipe = _recipe(cc, tmp_path, src)
    ensure_built(recipe)

    src.write_text("ERROR\n")
    result = ensure_built(recipe)
    assert isinstance(result, BuildError)
    assert "compile error" in result.msg
    assert not recipe.stamp.exists()

    # The old output is still there, but it must not be reused.
    src.write_text("a\n")
    ensure_built(recipe)
    assert builds.count == 3


def test_out_dir_is_emptied(tmp_path: Path) -> None:
    write_tree(tmp_path, {"a.c": "a\n", "classes": {"Stale.class": ""}})
    src = tmp_path / "a.c"
    out = tmp_path / "classes"
    recipe = BuildRecipe(
        cmd=[sys.executable, "-c", f"open({str(out / 'New.class')!r}, 'w')"],
        out=out,
        stamp=tmp_path / "classes.stamp.json",
        inputs=[src],
        out_is_dir=True,
    )

    assert ensure_built(recipe) is None
    assert sorted(p.name for p in out.iterdir()) == ["New.class"]


def test_parse_depfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Build an absolute path portably: `/abs/c.h` isn't absolute on Windows, which needs a drive.
    outside = tmp_path.parent / "c.h"
    escaped_outside = outside.as_posix().replace(" ", "\\ ")
    text = (
        "# env-dep:FOO=bar\n"
        "out.o: a.cpp inc/a\\ b.h \\\n"
        f"  {escaped_outside} $$d.h\n"
        "\n"
        "g.o : a.cpp g.cpp\n"
        "phony.rs:\n"
    )
    assert parse_depfile(text) == [
        tmp_path / "a.cpp",
        tmp_path / "inc" / "a b.h",
        outside,
        tmp_path / "$d.h",
        tmp_path / "g.cpp",
    ]
