"""
    ocimatic
    ~~~~~~~~
    Ocimatic is a tool for automating tasks related to the creation of problems
    for the Chilean Olympiad in Informatics (OCI)
    :license: Beer-Ware, see LICENSE.rst for more details.
"""

__version__ = 'beta1'

from typing import Optional, TypedDict
from pathlib import Path


class Config(TypedDict):
    timeout: int
    last_blank_page: bool
    verbosity: int
    contest_root: Path


config: Config = {'timeout': 10, 'last_blank_page': True, 'verbosity': 0, 'contest_root': Path('/')}
