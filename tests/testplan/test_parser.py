from __future__ import annotations

import pytest

from ocimatic.testplan import Extends
from ocimatic.utils import Stn

from .harness import assert_parse_errors, assert_parses_ok


def test_valid_testplan() -> None:
    subtasks = assert_parses_ok(r"""
        [Subtask 1]
          @validator validation/validator.cpp
          sample ; copy statement/sample-*.in
          small ; echo 1 2
          rand ; gen_random.py 10 100
        [Subtask 2]
          @extends subtask 1
    """)

    header, items = subtasks[1]
    assert header.number == 2
    [extends] = items
    assert isinstance(extends, Extends)
    assert extends.stn == Stn(1)


@pytest.mark.xfail(
    raises=ValueError,
    reason="`Stn(0)` raises ValueError instead of the parser reporting a ParseError",
)
def test_extends_subtask_zero() -> None:
    assert_parse_errors(r"""
        [Subtask 1]
          @extends subtask 0
        #~                 ^ subtask number must be greater than or equal to 1
          @extends subtask 00
        #~                 ^^ subtask number must be greater than or equal to 1
          small ; echo 1
    """)
