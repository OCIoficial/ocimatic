import contextlib
import os
import shutil
import subprocess
import tempfile
import time as pytime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, overload

SIGNALS = {
    1: "SIGHUP",
    2: "SIGINT",
    3: "SIGQUIT",
    4: "SIGILL",
    5: "SIGTRAP",
    6: "SIGABRT",
    7: "SIGEMT",
    8: "SIGFPE",
    9: "SIGKILL",
    10: "SIGBUS",
    11: "SIGSEGV",
    12: "SIGSYS",
    13: "SIGPIPE",
    14: "SIGALRM",
    15: "SIGTERM",
    16: "SIGURG",
    17: "SIGSTOP",
    18: "SIGTSTP",
    19: "SIGCONT",
    20: "SIGCHLD",
    21: "SIGTTIN",
    22: "SIGTTOU",
    23: "SIGIO",
    24: "SIGXCPU",
    25: "SIGXFSZ",
    26: "SIGVTALRM",
    27: "SIGPROF",
    28: "SIGWINCH",
    29: "SIGINFO",
    30: "SIGUSR1",
    31: "SIGUSR2",
}


@dataclass
class RunSuccess:
    time: float
    stdout: str
    stderr: str


@dataclass
class RunError:
    msg: str
    stderr: str


@dataclass
class RunTLE:
    pass


RunResult = RunSuccess | RunTLE | RunError


class Runnable(ABC):
    @abstractmethod
    def cmd(self) -> list[str]:
        raise NotImplementedError(
            "Class %s doesn't implement cmd()" % (self.__class__.__name__)
        )

    @overload
    def run(
        self,
        in_path: Path | None = None,
        out_path: Path | None = None,
        args: list[str] = [],
    ) -> RunSuccess | RunError:
        ...

    @overload
    def run(
        self,
        in_path: Path | None = None,
        out_path: Path | None = None,
        args: list[str] = [],
        timeout: float | None = None,
    ) -> RunResult:
        ...

    def run(
        self,
        in_path: Path | None = None,
        out_path: Path | None = None,
        args: list[str] = [],
        timeout: float | None = None,
    ) -> RunResult:
        """Run binary redirecting standard input and output.

        Arguments:
        ---------
            in_path: Path to redirect stdin from. If None input is redirected from /dev/null.
            out_path: File to redirec stdout to. If None output is redirected to a temporary file.
            args: Arguments to pass to the runnable.
            timeout: Timeout for the process. If None, there's no timeout.
        """
        assert in_path is None or in_path.exists()
        with contextlib.ExitStack() as stack:
            if not in_path:
                in_path = Path(os.devnull)
            in_file = stack.enter_context(in_path.open("r"))

            stdout: TextIO
            if out_path is None:
                stdout = stack.enter_context(tempfile.TemporaryFile("w+"))
            else:
                stdout = stack.enter_context(out_path.open("w+"))

            cmd = self.cmd()
            cmd.extend(args)

            start = pytime.monotonic()
            try:
                complete = subprocess.run(
                    cmd,
                    timeout=timeout,
                    stdin=in_file,
                    stdout=stdout,
                    text=True,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return RunTLE()
            time = pytime.monotonic() - start
            ret = complete.returncode
            status = ret == 0
            if not status:
                if ret < 0:
                    sig = -ret
                    msg = "Execution killed with signal %d" % sig
                    if sig in SIGNALS:
                        msg += ": %s" % SIGNALS[sig]
                else:
                    msg = "Execution ended with error (return code %d)" % ret
                return RunError(msg=msg, stderr=complete.stderr)

            stdout.seek(0)
            return RunSuccess(time=time, stdout=stdout.read(), stderr=complete.stderr)
        assert False


class Binary(Runnable):
    def __init__(self, path: str | Path) -> None:
        self._path = path

    def cmd(self) -> list[str]:
        return [str(self._path)]

    def is_callable(self) -> bool:
        return shutil.which(self._path) is not None


class JavaClasses(Runnable):
    def __init__(self, classname: str, classes: Path) -> None:
        self._classname = classname
        self._classes = classes

    def cmd(self) -> list[str]:
        return ["java", "-cp", str(self._classes), self._classname]


class Python3(Runnable):
    def __init__(self, script: Path) -> None:
        self._script = script

    def cmd(self) -> list[str]:
        return ["python3", str(self._script)]
