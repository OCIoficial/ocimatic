from __future__ import annotations

from unittest.mock import ANY

from ocimatic.testplan import (
    Copy,
    Echo,
    Extends,
    GroupName,
    Position,
    Range,
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

    # Exact ranges: the LSP uses them for go-to-definition and "file not found" diagnostics.
    assert subtasks == [
        (
            SubtaskHeader(number=1, range=_range(0, 0, 11)),
            [
                Validator(
                    path=Token(
                        lexeme="validation/validator.cpp",
                        kind=TokenKind.Word,
                        range=_range(1, 13, 37),
                    ),
                    range=_range(1, 2, 37),
                ),
                Copy(
                    group=GroupName("sample"),
                    pattern="statement/sample-*.in",
                    range=_range(2, 2, 37),
                ),
                Echo(group=GroupName("small"), args=["1", "2"], range=_range(3, 2, 18)),
                Script(
                    group=GroupName("rand"),
                    cmd=Token(
                        lexeme="gen_random.py",
                        kind=TokenKind.Word,
                        range=_range(4, 9, 22),
                    ),
                    args=["10", "100"],
                    range=_range(4, 2, 29),
                ),
            ],
        ),
        (
            SubtaskHeader(number=2, range=_range(5, 0, 11)),
            [Extends(stn=Stn(1), range=_range(6, 2, 20))],
        ),
    ]


def _range(line: int, start: int, end: int) -> Range:
    return Range(
        start=Position(line=line, column=start),
        end=Position(line=line, column=end),
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


def test_unknown_directive() -> None:
    assert_parse_errors(r"""
        [Subtask 1]
          @extend subtask 1
        #~^^^^^^^ expected `@extends` or `@validator`
    """)


def test_item_before_subtask() -> None:
    assert_parse_errors(r"""
          sample ; echo 1
        #~^^^^^^^^^^^^^^^ unexpected item before first subtask
    """)


def test_unexpected_token() -> None:
    assert_parse_errors(r"""
        [Subtask 1]
           ;
        #~ ^ unexpected token `;`
    """)


def test_header_errors() -> None:
    assert_parse_errors(r"""
          [Subtask]
        #~        ^ expected a number
          [Subtask 1
        #~          ^ expected `]`
          [subtask 1]
        #~ ^^^^^^^ expected `Subtask`
          [Subtask 1] x
        #~            ^ expected end of line
    """)


def test_directive_errors() -> None:
    assert_parse_errors(r"""
        [Subtask 1]
          @validator
        #~          ^ expected a path
          @extends 1
        #~         ^ expected `subtask`
          @extends subtask x
        #~                 ^ expected a number
    """)


def test_command_errors() -> None:
    assert_parse_errors(r"""
        [Subtask 1]
          a.b ; echo 1
        #~^^^ invalid group name: `a.b`
          small echo 1
        #~      ^^^^ expected `;`
          small ;
        #~       ^ expected `copy`, `echo` or a generator script
          small ; foo.sh
        #~        ^^^^^^ invalid command `foo.sh`
          small ; echo "abc
        #~             ^^^^ unterminated string
    """)


def test_copy_expects_exactly_one_argument() -> None:
    assert_parse_errors(r"""
        [Subtask 1]
          small ; copy
        #~        ^^^^ the `copy` command expects exactly one argument
          small ; copy a b
        #~        ^^^^^^^^ the `copy` command expects exactly one argument
    """)


def test_command_arguments() -> None:
    subtasks = assert_parses_ok(r"""
        [Subtask 1]
          quoted ; echo "a b"
          escape ; echo "a\nb"
          words ; echo -5 abc
          comment ; echo 1 # trailing comment
          cpp ; gen.cpp 1 2
    """)

    assert subtasks == [
        (
            SubtaskHeader(number=1, range=ANY),
            [
                Echo(group=GroupName("quoted"), args=["a b"], range=ANY),
                Echo(group=GroupName("escape"), args=["a\nb"], range=ANY),
                Echo(group=GroupName("words"), args=["-5", "abc"], range=ANY),
                Echo(group=GroupName("comment"), args=["1"], range=ANY),
                Script(
                    group=GroupName("cpp"),
                    cmd=Token(lexeme="gen.cpp", kind=TokenKind.Word, range=ANY),
                    args=["1", "2"],
                    range=ANY,
                ),
            ],
        ),
    ]


def test_blank_and_comment_lines_are_skipped() -> None:
    subtasks = assert_parses_ok(r"""
        # leading comment

        [Subtask 1]

          # comment-only line
          small ; echo 1
    """)

    assert subtasks == [
        (
            SubtaskHeader(number=1, range=ANY),
            [Echo(group=GroupName("small"), args=["1"], range=ANY)],
        ),
    ]


def test_non_ascii_string() -> None:
    subtasks = assert_parses_ok(r"""
        [Subtask 1]
          small ; echo "Ñandú"
    """)

    assert subtasks == [
        (
            SubtaskHeader(number=1, range=ANY),
            [Echo(group=GroupName("small"), args=["Ñandú"], range=ANY)],
        ),
    ]


def test_escaped_quote_at_end_of_string() -> None:
    subtasks = assert_parses_ok(r"""
        [Subtask 1]
          small ; echo "a\""
    """)

    assert subtasks == [
        (
            SubtaskHeader(number=1, range=ANY),
            [Echo(group=GroupName("small"), args=['a"'], range=ANY)],
        ),
    ]


def test_string_escapes() -> None:
    subtasks = assert_parses_ok(r"""
        [Subtask 1]
          small ; echo "q\"q" "b\\s" "n\nn" "t\tt" "\\n"
    """)

    assert subtasks == [
        (
            SubtaskHeader(number=1, range=ANY),
            [
                Echo(
                    group=GroupName("small"),
                    args=['q"q', "b\\s", "n\nn", "t\tt", "\\n"],
                    range=ANY,
                ),
            ],
        ),
    ]


def test_invalid_escape_sequence() -> None:
    assert_parse_errors(r"""
        [Subtask 1]
          small ; echo "a\qb"
        #~               ^^ invalid escape sequence `\q`
          small ; echo "\x4"
        #~              ^^ expected two hex digits after `\x`
          small ; echo "\xzz"
        #~              ^^ expected two hex digits after `\x`
    """)


def test_hex_escapes() -> None:
    subtasks = assert_parses_ok(r"""
        [Subtask 1]
          small ; echo "\x41\x6a" "\x7E" "\x411" "\\x41"
    """)

    assert subtasks == [
        (
            SubtaskHeader(number=1, range=ANY),
            [
                Echo(
                    group=GroupName("small"),
                    # Exactly two digits are consumed, and an escaped backslash
                    # doesn't start an escape.
                    args=["Aj", "~", "A1", "\\x41"],
                    range=ANY,
                ),
            ],
        ),
    ]
