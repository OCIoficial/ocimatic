from __future__ import annotations

from pathlib import Path

import pytest

from .tree import Tree, read_tree, write_tree


def test_write_tree_round_trips(tmp_path: Path) -> None:
    tree: Tree = {
        "a.txt": "1 2\r\n",
        "bin": b"\xff\x00",
        "empty": {},
        "nested": {"dir": {"b.txt": "", "c.txt": "c\n"}},
    }
    write_tree(tmp_path, tree)
    assert read_tree(tmp_path) == tree


def test_write_tree_splits_slash_paths(tmp_path: Path) -> None:
    write_tree(tmp_path, {"a/b/c.txt": "c\n", "d": {"e/f": {}}})
    assert read_tree(tmp_path) == {"a": {"b": {"c.txt": "c\n"}}, "d": {"e": {"f": {}}}}


def test_write_empty_tree_creates_nothing(tmp_path: Path) -> None:
    write_tree(tmp_path / "root", {})
    assert not (tmp_path / "root").exists()


@pytest.mark.parametrize(
    "tree",
    [
        pytest.param({"a/b": "1", "a/c": "2"}, id="shared-directory"),
        pytest.param({"a": {"b": "1"}, "a/c": "2"}, id="nested-and-slash"),
        pytest.param({"a": "1", "a/b": "2"}, id="file-and-directory"),
        pytest.param({"a": "1", "a/": "2"}, id="trailing-slash-alias"),
        pytest.param({"a": "1", "./a": "2"}, id="dot-alias"),
        pytest.param({"d": {"a/b": "1", "a/c": "2"}}, id="inside-nested-tree"),
    ],
)
def test_write_tree_rejects_overlapping_paths(tmp_path: Path, tree: Tree) -> None:
    with pytest.raises(ValueError, match="overlap"):
        write_tree(tmp_path / "root", tree)
    assert not (tmp_path / "root").exists()


@pytest.mark.parametrize("name", ["", ".", "/abs", "../up", "a/../b"])
def test_write_tree_rejects_invalid_paths(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="invalid tree path"):
        write_tree(tmp_path / "root", {"d": {name: "x"}})
    assert not (tmp_path / "root").exists()
