from __future__ import annotations

import pytest

from ocimatic.result import Error
from ocimatic.dataset import Outcome
from ocimatic.solutions import ExpectedComment
from ocimatic.utils import Stn


def test_expected_comment() -> None:
    comment = ExpectedComment.parse("[st1=OK, st2 = WA ]")
    assert isinstance(comment, ExpectedComment)
    assert dict(comment.subtasks.items()) == {Stn(1): Outcome.OK, Stn(2): Outcome.WA}


@pytest.mark.parametrize("item", ["st1=OK junk", "junk st1=OK", "st1=OK=WA"])
def test_expected_comment_rejects_extra_text(item: str) -> None:
    result = ExpectedComment.parse(f"[{item}]")
    assert result == Error(
        f"Items must be specified in the format `st{{n}}=VAL`, got `{item}`",
    )
