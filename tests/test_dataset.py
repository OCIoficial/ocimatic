from __future__ import annotations

from pathlib import Path

import pytest

from ocimatic.dataset import Dataset, normalize_content
from ocimatic.utils import Stn

from .tree import write_tree


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(b"1 2\r\n3\r\n", b"1 2\n3\n", id="crlf"),
        pytest.param(b"1 2\r3\r", b"1 2\n3\n", id="lone-cr"),
        pytest.param(b"1 2\r\n3\r4", b"1 2\n3\n4\n", id="mixed"),
        pytest.param(b"\xef\xbb\xbf1 2\n", b"1 2\n", id="bom"),
        pytest.param(b"1 2  \n3\t\n", b"1 2\n3\n", id="trailing-whitespace"),
        pytest.param(b"1 2\n\n\n", b"1 2\n", id="trailing-empty-lines"),
        pytest.param(b"1 2", b"1 2\n", id="missing-final-newline"),
        pytest.param(b"1\n\n2\n", b"1\n\n2\n", id="inner-empty-line-kept"),
        pytest.param(b"1\f\n", b"1\f\n", id="other-control-characters-kept"),
        pytest.param(b"", b"\n", id="empty"),
        pytest.param(b"\n\n  \n", b"\n", id="only-blank-lines"),
    ],
)
def test_normalize_content(content: bytes, expected: bytes) -> None:
    assert normalize_content(content) == expected


def test_normalized_content_is_unchanged() -> None:
    content = b"1 2\n3\n"
    assert normalize_content(content) == content


def test_static_dataset_subtask_numbers(tmp_path: Path) -> None:
    # Files are skipped and directories are numbered in alphabetical order, so `st10` is subtask 2.
    write_tree(
        tmp_path,
        {"dataset": {"data.zip": b"", "st1": {}, "st2": {}, "st10": {}}},
    )

    dataset = Dataset(tmp_path / "dataset", None, [])

    assert dataset.subtasks() == {Stn(1), Stn(2), Stn(3)}
    assert [str(dataset.subtask(Stn(i))) for i in range(1, 4)] == ["st1", "st10", "st2"]
