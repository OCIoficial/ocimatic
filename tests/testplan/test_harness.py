"""Tests for the parser test harness itself.

These only exercise the harness's own logic
"""

from __future__ import annotations

import pytest

from ocimatic.testplan import ParseError, Position, Range

from .harness import (
    AnnotationError,
    assert_parse_errors,
    assert_parses_ok,
    check_errors,
    dedent_source,
    parse_annotations,
)


def _span(line: int, start: int, end: int) -> Range:
    return Range(
        start=Position(line=line, column=start),
        end=Position(line=line, column=end),
    )


# Annotation parsing


def test_annotation_column_width_and_message() -> None:
    src = "[Subtask 1]\n  foo bar\n#~    ^^^ some message\n"
    assert parse_annotations(src) == [(_span(1, 6, 9), "some message")]


def test_annotation_trailing_whitespace_is_ignored() -> None:
    src = "[Subtask 1]\n  foo\n#~  ^ msg   \n"
    assert parse_annotations(src) == [(_span(1, 4, 5), "msg")]


def test_stacked_annotations_apply_to_the_same_line() -> None:
    src = "[Subtask 1]\n  foo bar\n#~ ^ first\n#~     ^ second\n"
    assert parse_annotations(src) == [
        (_span(1, 3, 4), "first"),
        (_span(1, 7, 8), "second"),
    ]


def test_annotations_apply_to_the_closest_source_line() -> None:
    src = "[Subtask 1]\n  foo\n#~  ^ a\n  bar\n#~  ^ b\n"
    assert parse_annotations(src) == [
        (_span(1, 4, 5), "a"),
        (_span(3, 4, 5), "b"),
    ]


def test_plain_comments_are_not_annotations() -> None:
    src = "[Subtask 1]\n# ^ not an annotation\n  # also not\n"
    assert parse_annotations(src) == []


# Dedent alignment


def test_dedent_keeps_carets_aligned() -> None:
    indented = dedent_source(r"""
        [Subtask 1]
          foo bar
        #~    ^^^ msg
    """)
    assert indented == "[Subtask 1]\n  foo bar\n#~    ^^^ msg\n"
    assert parse_annotations(indented) == [(_span(1, 6, 9), "msg")]


# Comparing errors


def test_check_errors_passes_on_match() -> None:
    check_errors(
        [(_span(1, 4, 8), "boom")],
        [ParseError(range=_span(1, 4, 8), msg="boom")],
    )


def test_check_errors_compares_only_the_first_line_of_the_message() -> None:
    check_errors(
        [(_span(1, 4, 8), "boom")],
        [ParseError(range=_span(1, 4, 8), msg="boom\nsome extra hint")],
    )


def test_check_errors_ignores_order() -> None:
    check_errors(
        [(_span(3, 2, 3), "second"), (_span(1, 4, 8), "first")],
        [
            ParseError(range=_span(1, 4, 8), msg="first"),
            ParseError(range=_span(3, 2, 3), msg="second"),
        ],
    )


def test_check_errors_fails_on_missing_error() -> None:
    with pytest.raises(pytest.fail.Exception, match=r"(?m)^-2:5-2:9: boom$"):
        check_errors([(_span(1, 4, 8), "boom")], [])


def test_check_errors_fails_on_unexpected_error() -> None:
    with pytest.raises(pytest.fail.Exception, match=r"(?m)^\+2:5-2:9: boom$"):
        check_errors([], [ParseError(range=_span(1, 4, 8), msg="boom")])


def test_check_errors_fails_on_misplaced_error() -> None:
    with pytest.raises(pytest.fail.Exception) as exc_info:
        check_errors(
            [(_span(1, 5, 9), "boom")],
            [ParseError(range=_span(1, 4, 8), msg="boom")],
        )
    output = str(exc_info.value)
    assert "\n-2:6-2:10: boom" in output
    assert "\n+2:5-2:9: boom" in output


def test_check_errors_fails_on_wrong_message() -> None:
    with pytest.raises(pytest.fail.Exception) as exc_info:
        check_errors(
            [(_span(1, 4, 8), "boom")],
            [ParseError(range=_span(1, 4, 8), msg="bang")],
        )
    output = str(exc_info.value)
    assert "\n-2:5-2:9: boom" in output
    assert "\n+2:5-2:9: bang" in output


def test_check_errors_reports_errors_without_range() -> None:
    with pytest.raises(pytest.fail.Exception, match=r"(?m)^\+<no range>: cycles$"):
        check_errors([], [ParseError(msg="cycles")])


# Malformed annotations


@pytest.mark.parametrize(
    "src",
    [
        pytest.param("#~ ^ msg\n[Subtask 1]\n", id="first-line"),
        pytest.param("[Subtask 1]\n#~ msg\n", id="no-carets"),
        pytest.param("[Subtask 1]\n#~   ^\n", id="no-message"),
        pytest.param("[Subtask 1]\n  foo\n  #~ ^ msg\n", id="indented-marker"),
    ],
)
def test_malformed_annotation(src: str) -> None:
    with pytest.raises(AnnotationError):
        parse_annotations(src)


def test_assert_parses_ok_rejects_annotations() -> None:
    with pytest.raises(AnnotationError):
        assert_parses_ok("[Subtask 1]\n  small ; echo 1\n#~        ^ msg\n")


def test_assert_parse_errors_requires_annotations() -> None:
    with pytest.raises(AnnotationError):
        assert_parse_errors("[Subtask 1]\n  small ; echo 1\n")
