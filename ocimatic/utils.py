from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Generic, Protocol, TypeVar

import ocimatic


def relative_to_cwd(path: Path) -> str:
    commonpath = Path(os.path.commonpath([path, Path.cwd()]))
    if commonpath.is_relative_to(ocimatic.contest_root):
        relpath = os.path.relpath(path, Path.cwd())
        if not relpath.startswith("."):
            relpath = "." + os.path.sep + relpath
        return relpath
    else:
        return str(path)


_T = TypeVar("_T")


class Comparable(Protocol):
    def __lt__(self: _T, other: _T) -> bool: ...


_K = TypeVar("_K", bound=Comparable)
_V = TypeVar("_V")


class SortedDict(Generic[_K, _V]):
    """A dict that iterates over keys in sorted order."""

    def __init__(self, iter: Iterable[tuple[_K, _V]] | None = None) -> None:
        self._dict = dict(iter or [])

    def __getitem__(self, key: _K) -> _V:
        return self._dict[key]

    def __setitem__(self, key: _K, val: _V) -> None:
        self._dict[key] = val

    def __contains__(self, key: _K) -> bool:
        return key in self._dict

    def __repr__(self) -> str:
        items = ", ".join(f"{key!r}: {val!r}" for key, val in self.items())
        return f"{{{items}}}"

    def __len__(self) -> int:
        return len(self._dict)

    def setdefault(self, key: _K, default: _V) -> _V:
        return self._dict.setdefault(key, default)

    def keys(self) -> list[_K]:
        return sorted(self._dict.keys())

    def values(self) -> Iterator[_V]:
        for key in self.keys():
            yield self._dict[key]

    def items(self) -> Iterator[tuple[_K, _V]]:
        for key in self.keys():
            yield (key, self._dict[key])

    def __iter__(self) -> Iterator[_K]:
        yield from self.keys()


class Stn:
    """A wrapper over an integer used as an identifier for a subtask."""

    def __init__(self, stn: int) -> None:
        if stn < 1:
            raise ValueError("Subtask number must be greater than or equal to 1")
        self._idx = stn

    def __hash__(self) -> int:
        return hash(self._idx)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stn):
            raise ValueError(f"Cannot compare Stn with {type(other)}")
        return self._idx == other._idx

    def __str__(self) -> str:
        return f"{self._idx}"

    def __repr__(self) -> str:
        return f"st{self._idx}"

    def __lt__(self, other: Stn) -> bool:
        return self._idx < other._idx
