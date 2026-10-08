from __future__ import annotations

import pytest

from ocimatic.testplan import ParseError, Position, Range, Token, TokenKind, tokenize


def _token(kind: TokenKind, lexeme: str, start: int, value: str | None = None) -> Token:
    """Return a token on line 0 starting at column `start`."""
    return Token(
        kind=kind,
        lexeme=lexeme,
        range=_range(start, start + len(lexeme)),
        value=value,
    )


def _eol(line: str) -> Token:
    return Token(kind=TokenKind.Eol, lexeme="", range=_range(len(line), len(line) + 1))


def _range(start: int, end: int) -> Range:
    return Range(start=Position(line=0, column=start), end=Position(line=0, column=end))


def test_header() -> None:
    line = "[Subtask 1]"
    assert tokenize(0, line) == [
        _token(TokenKind.OpenBracket, "[", 0),
        _token(TokenKind.Word, "Subtask", 1),
        _token(TokenKind.Num, "1", 9),
        _token(TokenKind.CloseBracket, "]", 10),
        _eol(line),
    ]


def test_directive() -> None:
    line = "  @extends subtask 12"
    assert tokenize(0, line) == [
        _token(TokenKind.Directive, "@extends", 2),
        _token(TokenKind.Word, "subtask", 11),
        _token(TokenKind.Num, "12", 19),
        _eol(line),
    ]


def test_command() -> None:
    line = '  rand-1 ; gen.py -5 "a b" ./x*.in 7 ; # trailing comment'
    assert tokenize(0, line) == [
        _token(TokenKind.Word, "rand-1", 2),
        _token(TokenKind.Semicolon, ";", 9),
        _token(TokenKind.Word, "gen.py", 11),
        _token(TokenKind.Word, "-5", 18),
        _token(TokenKind.String, '"a b"', 21, value="a b"),
        _token(TokenKind.Word, "./x*.in", 27),
        _token(TokenKind.Num, "7", 35),
        _token(TokenKind.Semicolon, ";", 37),
        _eol(line),
    ]


def test_token_line_number() -> None:
    tokens = tokenize(3, "word")
    assert not isinstance(tokens, ParseError)
    [token, _] = tokens
    assert token.range == Range(
        start=Position(line=3, column=0),
        end=Position(line=3, column=4),
    )


@pytest.mark.parametrize("line", ["", "   ", "# comment", "  # comment"])
def test_blank_lines(line: str) -> None:
    assert tokenize(0, line) == [_eol(line)]


def test_string_values() -> None:
    line = r'"" "x # y" "q\"q" "b\\s" "n\nn" "t\tt" "\x41\x6a" "\\x41" "ñ"'
    tokens = tokenize(0, line)
    assert not isinstance(tokens, ParseError)
    assert [t.value for t in tokens if t.kind == TokenKind.String] == [
        "",
        "x # y",
        'q"q',
        "b\\s",
        "n\nn",
        "t\tt",
        "Aj",
        "\\x41",
        "ñ",
    ]


def test_adjacent_tokens() -> None:
    # A string doesn't join the word next to it.
    line = 'ab"c d"e'
    assert tokenize(0, line) == [
        _token(TokenKind.Word, "ab", 0),
        _token(TokenKind.String, '"c d"', 2, value="c d"),
        _token(TokenKind.Word, "e", 7),
        _eol(line),
    ]


@pytest.mark.parametrize("char", ["$", "\\", "!", "{"])
def test_other_characters_are_error_tokens(char: str) -> None:
    line = f"a {char} b"
    assert tokenize(0, line) == [
        _token(TokenKind.Word, "a", 0),
        _token(TokenKind.Error, char, 2),
        _token(TokenKind.Word, "b", 4),
        _eol(line),
    ]


@pytest.mark.parametrize(
    ("line", "msg", "start", "end"),
    [
        pytest.param('"abc', "unterminated string", 0, 4, id="unterminated"),
        pytest.param(
            '"abc\\"',
            "unterminated string",
            0,
            6,
            id="unterminated-escaped-quote",
        ),
        pytest.param(
            '"abc\\',
            "unterminated string",
            0,
            5,
            id="unterminated-backslash",
        ),
        pytest.param(
            'a "b" "c',
            "unterminated string",
            6,
            8,
            id="unterminated-after-string",
        ),
        pytest.param(
            '"a\\qb"',
            "invalid escape sequence `\\q`",
            2,
            4,
            id="invalid-escape",
        ),
        pytest.param(
            '"\\x4"',
            "expected two hex digits after `\\x`",
            1,
            3,
            id="short-hex",
        ),
        pytest.param(
            '"\\xzz"',
            "expected two hex digits after `\\x`",
            1,
            3,
            id="invalid-hex",
        ),
    ],
)
def test_lexical_errors(line: str, msg: str, start: int, end: int) -> None:
    assert tokenize(0, line) == ParseError(msg=msg, range=_range(start, end))
