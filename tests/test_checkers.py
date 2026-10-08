from __future__ import annotations

from pathlib import Path

import pytest

from ocimatic.checkers import CheckerSuccess, DiffChecker


def _check(tmp_path: Path, out: bytes, expected: bytes) -> CheckerSuccess:
    (tmp_path / "test.in").write_bytes(b"")
    (tmp_path / "test.out").write_bytes(out)
    (tmp_path / "test.sol").write_bytes(expected)
    result = DiffChecker().run(
        in_path=tmp_path / "test.in",
        expected_path=tmp_path / "test.sol",
        out_path=tmp_path / "test.out",
    )
    assert isinstance(result, CheckerSuccess)
    return result


@pytest.mark.parametrize(
    ("out", "expected"),
    [
        pytest.param(b"1 2\n3\n", b"1 2\n3\n", id="identical"),
        pytest.param(b"  1 \t 2\x0b\x0c\n", b"1 2\n", id="whitespace-runs"),
        pytest.param(b"1 2\r\n3\r\n", b"1 2\n3\n", id="crlf"),
        pytest.param(b"1\n3\n\n \n", b"1\n3", id="trailing-blank-lines"),
        pytest.param(b"\xff 1\n", b"\xff 1\n", id="invalid-utf8"),
    ],
)
def test_accepts(tmp_path: Path, out: bytes, expected: bytes) -> None:
    assert _check(tmp_path, out, expected) == CheckerSuccess(outcome=1.0)


@pytest.mark.parametrize(
    ("out", "expected", "msg"),
    [
        pytest.param(
            b"1 3\n",
            b"1 2\n",
            "Expected `1 2`, found `1 3` on line 1",
            id="wrong",
        ),
        pytest.param(b"1\n2\n", b"1\n", "Contestant output too long", id="too-long"),
        pytest.param(b"1\n", b"1\n2\n", "Contestant output too short", id="too-short"),
        # Only `\n` ends a line; `\r` is whitespace.
        pytest.param(
            b"1\r2\n",
            b"1\n2\n",
            "Expected `1`, found `1 2` on line 1",
            id="lone-cr",
        ),
        # Blank lines in the middle count.
        pytest.param(
            b"1\n\n2\n",
            b"1\n2\n",
            "Expected `2`, found `` on line 2",
            id="inner-blank",
        ),
        # Non-ASCII whitespace isn't whitespace.
        pytest.param(
            b"1\xc2\xa02\n",
            b"1 2\n",
            "Expected `1 2`, found `1\xa02` on line 1",
            id="nbsp",
        ),
    ],
)
def test_rejects(tmp_path: Path, out: bytes, expected: bytes, msg: str) -> None:
    assert _check(tmp_path, out, expected) == CheckerSuccess(outcome=0.0, msg=msg)
