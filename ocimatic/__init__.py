"""
    ocimatic
    ~~~~~~~~
    Ocimatic is a tool for automating tasks related to the creation of problems
    for the Chilean Olympiad in Informatics (OCI)
    :license: Beer-Ware, see LICENSE.rst for more details.
"""

__version__ = '0.1.0'

from pathlib import Path
from typing import TypedDict


class Config(TypedDict):
    last_blank_page: bool
    verbosity: int
    contest_root: Path


config: Config = {'last_blank_page': True, 'verbosity': 1, 'contest_root': Path('/')}
