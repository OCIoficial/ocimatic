"""Helpers for writing multi-line text in tests."""

from __future__ import annotations

import textwrap


def block(text: str) -> str:
    """Dedent an indented triple-quoted string, dropping the newline right after the opening quotes.

    Write the opening quotes at the end of a line and the content indented on the following lines.
    The line with the closing quotes only contains whitespace, so the result ends with a single
    newline.
    """
    return textwrap.dedent(text.removeprefix("\n"))
