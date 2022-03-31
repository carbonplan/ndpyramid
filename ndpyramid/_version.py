# flake8: noqa
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('ndpyramid')
except PackageNotFoundError:
    # package is not installed
    __version__ = 'unknown'
