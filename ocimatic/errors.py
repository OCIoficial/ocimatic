"""Errors reported to the user."""

from __future__ import annotations

from typing import IO, Any

import click

from ocimatic import ui


class OcimaticError(click.ClickException):
    """An error caused by the user's input or environment, reported without a traceback.

    When raised while running a command, click calls `show` to print the message and exits with
    code 1. `details` is optional longer text (e.g. a list of errors) shown below the message.
    """

    def __init__(self, message: str, *, details: str | None = None) -> None:
        super().__init__(message)
        self.details = details

    def show(self, file: IO[Any] | None = None) -> None:
        del file
        ui.writeln(ui.colorize(self.message, ui.INFO + ui.RED))
        if self.details:
            ui.dump_message(self.details)
        ui.writeln()
