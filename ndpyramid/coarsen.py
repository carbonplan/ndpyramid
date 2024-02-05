from __future__ import annotations  # noqa: F401

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

    # multiscales spec
    save_kwargs = locals()
    del save_kwargs['ds']

    attrs = {
        'multiscales': multiscales_template(
            datasets=[{'path': str(i)} for i in range(len(factors))],
            type='reduce',
            method='pyramid_coarsen',
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    # set up pyramid
    plevels = {}

    # pyramid data
    for key, factor in enumerate(factors):
        # merge dictionary via union operator
        kwargs |= {d: factor for d in dims}
        plevels[str(key)] = ds.coarsen(**kwargs).mean()  # type: ignore

    plevels['/'] = xr.Dataset(attrs=attrs)
    return dt.DataTree.from_dict(plevels)
