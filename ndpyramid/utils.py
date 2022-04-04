from __future__ import annotations  # noqa: F401

from ._version import __version__


def get_version() -> str:
    return __version__


def multiscales_template(
    datasets: list = None,
    type: str = '',
    method: str = '',
    version: str = '',
    args: list = None,
    kwargs: dict = None,
):
    if datasets is None:
        datasets = []
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    # https://forum.image.sc/t/multiscale-arrays-v0-1/37930
    return [
        {
            'datasets': datasets,
            'type': type,
            'metadata': {'method': method, 'version': version, 'args': args, 'kwargs': kwargs},
        }
    ]
