from __future__ import annotations

from unittest.mock import ANY

import pytest

from ocimatic.testplan import (
    Copy,
    Echo,
    Extends,
    GroupName,
    Script,
    SubtaskHeader,
    Token,
    TokenKind,
    Validator,
)
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

    # Ranges are wildcarded with `ANY`, so this only checks the parsed structure.
    assert subtasks == [
        (
            SubtaskHeader(number=1, range=ANY),
            [
                Validator(
                    path=Token(
                        lexeme="validation/validator.cpp",
                        kind=TokenKind.Word,
                        range=ANY,
                    ),
                    range=ANY,
                ),
                Copy(
                    group=GroupName("sample"),
                    pattern="statement/sample-*.in",
                    range=ANY,
                ),
                Echo(group=GroupName("small"), args=["1", "2"], range=ANY),
                Script(
                    group=GroupName("rand"),
                    cmd=Token(lexeme="gen_random.py", kind=TokenKind.Word, range=ANY),
                    args=["10", "100"],
                    range=ANY,
                ),
            ],
        ),
        (
            SubtaskHeader(number=2, range=ANY),
            [Extends(stn=Stn(1), range=ANY)],
        ),
    ]


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
