from __future__ import annotations  # noqa: F401

from collections.abc import Callable

import xarray as xr

from .utils import get_version, multiscales_template


def pyramid_create(
    ds: xr.Dataset,
    *,
    factors: list[int],
    dims: list[str],
    func: Callable,
    type_label: str = "reduce",
    method_label: str | None = None,
    **kwargs,
):
    """Create a multiscale pyramid via a given function applied to a dataset.
    The generalized version of pyramid_coarsen.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to apply the function to.
    factors : list[int]
        The factors to coarsen by.
    dims : list[str]
        The dimensions to coarsen.
    func : Callable
        The function to apply to the dataset; must accept the
        `ds`, `factor`, and `dims` as positional arguments.
    type_label : str, optional
        The type label to use as metadata for the multiscales spec.
        The default is 'reduce'.
    method_label : str, optional
        The method label to use as metadata for the multiscales spec.
        The default is the name of the function.
    kwargs : dict
        Additional keyword arguments to pass to the func.

    """
    # multiscales spec
    save_kwargs = locals()
    del save_kwargs["ds"]
    del save_kwargs["func"]
    del save_kwargs["type_label"]
    del save_kwargs["method_label"]

    attrs = {
        "multiscales": multiscales_template(
            datasets=[{"path": str(i)} for i in range(len(factors))],
            type=type_label,
            method=method_label or func.__name__,
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    # set up pyramid
    plevels = {}

    # pyramid data
    for key, factor in enumerate(factors):
        plevels[str(key)] = func(ds, factor, dims, **kwargs)

    plevels["/"] = xr.Dataset(attrs=attrs)
    return xr.DataTree.from_dict(plevels)
