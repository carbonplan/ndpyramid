from pkg_resources import DistributionNotFound, get_distribution

from .core import pyramid_coarsen, pyramid_reproject

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # noqa: F401; pragma: no cover
    # package is not installed
    __version__ = '-9999'
