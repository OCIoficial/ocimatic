"""The environment a command runs in.

Everything that would otherwise be global state (configuration, working directory, contest root and
output verbosity) lives in an `Env`. An environment is installed with `Env.use` and queried from
anywhere with `Env.get`.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Generator
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ocimatic.config import Config


class Verbosity(Enum):
    quiet = 0
    verbose = 2


@dataclass(frozen=True, kw_only=True, slots=True)
class Env:
    config: Config
    cwd: Path
    contest_root: Path | None
    """Root of the contest containing `cwd`, or `None` when not inside a contest."""

    verbosity: Verbosity = Verbosity.verbose

    def require_contest_root(self) -> Path:
        """Return the contest root, failing if not inside a contest."""
        from ocimatic.errors import OcimaticError

        if self.contest_root is None:
            raise OcimaticError("ocimatic was not called inside a contest.")
        return self.contest_root

    @staticmethod
    def get() -> Env:
        """Return the installed environment."""
        try:
            return _current.get()
        except LookupError:
            raise RuntimeError(
                "no environment installed; wrap the call in `Env.use(...)`",
            ) from None

    @staticmethod
    @contextlib.contextmanager
    def use(env: Env) -> Generator[Env]:
        """Install `env` for the duration of the block, restoring the previous one afterwards."""
        token = _current.set(env)
        try:
            yield env
        finally:
            _current.reset(token)

    @staticmethod
    @contextlib.contextmanager
    def override(**changes: Any) -> Generator[Env]:
        """Install a copy of the current environment with some fields changed."""
        with Env.use(dataclasses.replace(Env.get(), **changes)) as env:
            yield env


_current: ContextVar[Env] = ContextVar("ocimatic_env")
