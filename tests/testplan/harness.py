"""Helpers for writing declarative testplan parser tests.

A test input is a testplan written inline as a raw triple-quoted string, which is passed through
`block` to remove its indentation. Expected errors are annotated below the offending line with
`#~`, followed by carets marking the columns the error spans and the first line of the error
message:

    assert_parse_errors(r'''
        [Subtask 1]
          @extends subtask 0
        #~                 ^ subtask number must be greater than or equal to 1
    ''')

Annotations are ordinary comments for the parser, so they don't change how the input parses.
Because `#~` occupies columns 0 and 1, an error must start at column 2 or later to be annotated;
indent the source line if needed (testplans allow leading whitespace).
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable

import pytest

from ocimatic.testplan import Item, Parser, ParseError, Position, Range, SubtaskHeader

from ..text import block

MARKER = "#~"
_ANNOTATION_RE = re.compile(r"#~(\s*)(\^+)\s*(\S.*)")

type Annotation = tuple[Range, str]
"""An expected error: the range it spans and the first line of its message."""


class AnnotationError(Exception):
    """An annotation in a test input is malformed. This is a bug in the test, not the parser."""


def parse_annotations(src: str) -> list[Annotation]:
    """Extract the expected errors from `src`.

    Each annotation applies to the closest line above it that isn't an annotation.
    """
    annotations: list[Annotation] = []
    target: int | None = None
    for lineno, line in enumerate(src.splitlines()):
        if not line.lstrip().startswith(MARKER):
            target = lineno
            continue

        where = f"line {lineno + 1}: {line!r}"
        if not line.startswith(MARKER):
            raise AnnotationError(
                f"`{MARKER}` must be at the start of the line, {where}",
            )
        if target is None:
            raise AnnotationError(f"annotation has no source line above it, {where}")
        m = _ANNOTATION_RE.fullmatch(line.rstrip())
        if not m:
            raise AnnotationError(
                f"expected `{MARKER}` followed by carets and a message, {where}",
            )

        range = Range(
            start=Position(line=target, column=m.start(2)),
            end=Position(line=target, column=m.end(2)),
        )
        annotations.append((range, m.group(3)))
    return annotations


def assert_parses_ok(src: str) -> list[tuple[SubtaskHeader, list[Item]]]:
    """Assert that `src` parses without errors and return the parsed subtasks.

    `src` must not contain annotations.
    """
    src = block(src)
    if parse_annotations(src):
        raise AnnotationError(
            "`assert_parses_ok` expects no annotations, use `assert_parse_errors` instead",
        )
    parser = _parse(src)
    check_errors([], parser.errors)
    return parser.subtasks


def assert_parse_errors(src: str) -> None:
    """Assert that parsing `src` produces exactly the errors annotated in it.

    `src` must contain at least one annotation.
    """
    src = block(src)
    expected = parse_annotations(src)
    if not expected:
        raise AnnotationError(
            "`assert_parse_errors` expects at least one annotation, use `assert_parses_ok` instead",
        )
    parser = _parse(src)
    check_errors(expected, parser.errors)


def _parse(src: str) -> Parser:
    parser = Parser()
    parser.parse(src)
    return parser


def check_errors(expected: Iterable[Annotation], errors: Iterable[ParseError]) -> None:
    """Fail the test with a diff unless `errors` match the `expected` annotations exactly.

    Only the first line of each error message is compared.
    """
    actual = [(err.range, err.msg.split("\n", 1)[0]) for err in errors]
    expected_lines = _render(expected)
    actual_lines = _render(actual)
    if expected_lines != actual_lines:
        diff = difflib.unified_diff(
            expected_lines,
            actual_lines,
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
        pytest.fail(
            "parse errors don't match the annotations\n" + "\n".join(diff),
            pytrace=False,
        )


def _render(errors: Iterable[tuple[Range | None, str]]) -> list[str]:
    """Render errors one per line, sorted by position, in the format used by the CLI."""

    def key(error: tuple[Range | None, str]) -> tuple[int, int, int, int, int, str]:
        range, msg = error
        if range is None:
            return (0, 0, 0, 0, 0, msg)
        return (
            1,
            range.start.line,
            range.start.column,
            range.end.line,
            range.end.column,
            msg,
        )

    return [
        f"{range if range else '<no range>'}: {msg}"
        for range, msg in sorted(errors, key=key)
    ]
