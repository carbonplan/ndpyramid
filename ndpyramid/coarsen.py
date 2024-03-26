from __future__ import annotations  # noqa: F401

from typing import Callable

import datatree as dt
import xarray as xr

from .utils import get_version, multiscales_template


def pyramid_coarsen(
    ds: xr.Dataset, *, factors: list[int], dims: list[str], **kwargs
) -> dt.DataTree:
    """Create a multiscale pyramid via coarsening of a dataset by given factors

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to coarsen.
    factors : list[int]
        The factors to coarsen by.
    dims : list[str]
        The dimensions to coarsen.
    kwargs : dict
        Additional keyword arguments to pass to xarray.Dataset.coarsen.
    """

    def coarsen(ds: xr.Dataset, factor: int, **kwargs):
        # merge dictionary via union operator
        kwargs |= {d: factor for d in dims}
        return ds.coarsen(**kwargs).mean()  # type: ignore

    return pyramid_create(
        ds,
        factors=factors,
        dims=dims,
        func=coarsen,
        method_label="pyramid_coarsen",
        type_label="reduce",
        **kwargs,
    )


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
    del save_kwargs['ds']

    attrs = {
        'multiscales': multiscales_template(
            datasets=[{'path': str(i)} for i in range(len(factors))],
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

    plevels['/'] = xr.Dataset(attrs=attrs)
    return dt.DataTree.from_dict(plevels)



# 

"""

"""