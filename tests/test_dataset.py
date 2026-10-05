from __future__ import annotations

import pytest

from ocimatic.dataset import normalize_content


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
