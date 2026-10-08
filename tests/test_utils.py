from __future__ import annotations

from unittest.mock import ANY

from ocimatic.utils import Stn


def test_stn_equality() -> None:
    assert Stn(1) == Stn(1)
    assert Stn(1) != Stn(2)


def test_stn_compares_unequal_to_other_types() -> None:
    assert Stn(1) != None  # noqa: E711
    assert Stn(1) != 1
    assert Stn(1) == ANY
