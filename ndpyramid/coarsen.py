from __future__ import annotations  # noqa: F401

import xarray as xr

from .create import pyramid_create


def pyramid_coarsen(
    ds: xr.Dataset, *, factors: list[int], dims: list[str], **kwargs
) -> xr.DataTree:
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

    def coarsen(ds: xr.Dataset, factor: int, dims: list[str], **kwargs):
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
