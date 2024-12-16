"""Ocimatic is a tool for automating tasks related to the creation of problems for the Chilean Olympiad in Informatics (OCI).

:license: Beer-Ware, see LICENSE.rst for more details.
"""

from __future__ import annotations


def main() -> None:
    from ocimatic.main import cli

    cli()


if __name__ == "__main__":
    main()
