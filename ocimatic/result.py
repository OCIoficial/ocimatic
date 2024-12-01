from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Status(Enum):
    """A boolean-like enum to signal whether some process was succseful or failed."""

    success = "success"
    fail = "fail"

    @staticmethod
    def from_bool(b: bool) -> Status:  # noqa: FBT001
        return Status.success if b else Status.fail

    def to_bool(self) -> bool:
        match self:
            case Status.success:
                return True
            case Status.fail:
                return False

    def __iand__(self, other: Status) -> Status:
        return Status.from_bool(self.to_bool() and other.to_bool())


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkResult:
    status: Status | None
    short_msg: str
    long_msg: str | None = None

    @staticmethod
    def success(short_msg: str, long_msg: str | None = None) -> WorkResult:
        return WorkResult(status=Status.success, short_msg=short_msg, long_msg=long_msg)

    @staticmethod
    def fail(short_msg: str, long_msg: str | None = None) -> WorkResult:
        return WorkResult(status=Status.fail, short_msg=short_msg, long_msg=long_msg)

    @staticmethod
    def info(short_msg: str, long_msg: str | None = None) -> WorkResult:
        return WorkResult(status=None, short_msg=short_msg, long_msg=long_msg)

    def into_work_result(self) -> WorkResult:
        return self


class IntoWorkResult(Protocol):
    def into_work_result(self) -> WorkResult: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class Result:
    status: Status
    short_msg: str
    long_msg: str | None = None

    @staticmethod
    def success(short_msg: str, long_msg: str | None = None) -> Result:
        return Result(status=Status.success, short_msg=short_msg, long_msg=long_msg)

    @staticmethod
    def fail(short_msg: str, long_msg: str | None = None) -> Result:
        return Result(status=Status.fail, short_msg=short_msg, long_msg=long_msg)

    def is_fail(self) -> bool:
        return self.status == Status.fail

    def into_work_result(self) -> WorkResult:
        return WorkResult(
            status=self.status,
            short_msg=self.short_msg,
            long_msg=self.long_msg,
        )


@dataclass(frozen=True, slots=True)
class Error:
    """A generic error carrying a message.

    This is typically used as part of a union in a return type to signal that a function
    can return some value or fail with a message.
    """

    msg: str
