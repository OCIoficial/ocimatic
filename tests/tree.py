r"""Write and read file trees in tests.

A tree maps paths to contents: `str` or `bytes` for a file, and another tree for a directory, e.g.
`{"st1": {"a.in": "1 2\n", "a.sol": "3\n"}, "st2": {}}`. Paths are relative and `/`-separated, so
`{"st1/a.in": "1 2\n"}` is also a tree. Entries in the same tree can't overlap: the first components
of their paths must differ, so files sharing a directory are written as one nested tree.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath

type Tree = Mapping[str, str | bytes | Tree]


def write_tree(root: Path, tree: Tree) -> None:
    r"""Write `tree` under `root`.

    `str` is written as UTF-8 with `\n` kept as is, `bytes` is written as is, and a tree creates a
    directory, so `{}` is an empty directory. Only the entries are written: an empty tree doesn't
    create `root`. The whole tree is validated before anything is written.

    Raises:
        ValueError: If a path isn't relative, is empty, contains `..`, or overlaps another path.

    """
    _validate(tree, PurePosixPath())
    _write(root, tree)


def write_file(path: Path, content: str | bytes) -> None:
    """Write `content` to `path`, creating its parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        # `newline=""` writes `\n` as is; otherwise Windows would turn it into `\r\n`.
        path.write_text(content, encoding="utf-8", newline="")


def read_tree(root: Path) -> Tree:
    """Read the files and directories under `root` into a tree with one path component per key.

    Files that decode as UTF-8 are read as `str` and other files as `bytes`.
    """
    tree: dict[str, str | bytes | Tree] = {}
    for path in sorted(root.iterdir()):
        if path.is_dir():
            tree[path.name] = read_tree(path)
            continue
        content = path.read_bytes()
        try:
            tree[path.name] = content.decode("utf-8")
        except UnicodeDecodeError:
            tree[path.name] = content
    return tree


def _validate(tree: Tree, parent: PurePosixPath) -> None:
    seen: dict[str, str] = {}
    for name, content in tree.items():
        parts = _parts(name, parent)
        if parts[0] in seen:
            raise ValueError(
                f"tree paths `{parent / seen[parts[0]]}` and `{parent / name}` overlap; "
                f"write the entries under `{parent / parts[0]}` as one nested tree",
            )
        seen[parts[0]] = name
        if isinstance(content, Mapping):
            _validate(content, parent.joinpath(*parts))


def _parts(name: str, parent: PurePosixPath) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(
            f"invalid tree path `{name}` in `{parent}`: it must be relative, non-empty and "
            "without `..`",
        )
    return path.parts


def _write(root: Path, tree: Tree) -> None:
    for name, content in tree.items():
        path = root.joinpath(*PurePosixPath(name).parts)
        if isinstance(content, Mapping):
            path.mkdir(parents=True, exist_ok=True)
            _write(path, content)
        else:
            write_file(path, content)
