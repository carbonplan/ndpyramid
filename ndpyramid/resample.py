from __future__ import annotations  # noqa: F401

import datatree as dt
import numpy as np
import xarray as xr

from .common import Projection, ProjectionOptions
from .utils import (
    add_metadata_and_zarr_encoding,
    get_levels,
    get_version,
    multiscales_template,
)


def _da_resample(da, *, dim, projection_model):
    if da.encoding.get('_FillValue') is None and np.issubdtype(da.dtype, np.floating):
        da.encoding['_FillValue'] = np.nan
    return da


def level_resample(
    ds: xr.Dataset,
    *,
    projection: ProjectionOptions = 'web-mercator',
    level: int,
    pixels_per_tile: int = 128,
    clear_attrs: bool = False,
) -> xr.Dataset:
    """Create a level of a multiscale pyramid of a dataset via resampling.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to create a multiscale pyramid of.
    projection : str, optional
        The projection to use. Default is 'web-mercator'.
    level : int
        The level of the pyramid to create.
    pixels_per_tile : int, optional
        Number of pixels per tile
    clear_attrs : bool, False
        Clear the attributes of the DataArrays within the multiscale level. Default is False.

    Returns
    -------
    xr.Dataset
        The multiscale pyramid level.

    Warning
    -------
    Pyramid generation by level is experimental and subject to change.
    """
    save_kwargs = {'pixels_per_tile': pixels_per_tile}
    attrs = {
        'multiscales': multiscales_template(
            datasets=[{'path': '.', 'level': level}],
            type='reduce',
            method='pyramid_resample',
            version=get_version(),
            kwargs=save_kwargs,
        )
    }
    dim = 2**level * pixels_per_tile
    projection_model = Projection(name=projection)

    # create the data array for each level
    ds_level = xr.Dataset(attrs=ds.attrs)
    for k, da in ds.items():
        if clear_attrs:
            da.attrs.clear()
        if len(da.shape) > 3:
            # if extra_dim is not specified, raise an error
            raise NotImplementedError(
                '4+ dimensional datasets are not currently supported for pyramid_resample.'
            )
        else:
            # if the data array is not 4D, just resample it
            ds_level[k] = _da_resample(da, dim=dim, projection_model=projection_model)
    ds_level.attrs['multiscales'] = attrs['multiscales']
    return ds_level


def pyramid_resample(
    ds: xr.Dataset,
    *,
    projection: ProjectionOptions = 'web-mercator',
    levels: int = None,
    pixels_per_tile: int = 128,
    other_chunks: dict = None,
    clear_attrs: bool = False,
) -> dt.DataTree:
    """Create a multiscale pyramid of a dataset via resampling.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to create a multiscale pyramid of.
    projection : str, optional
        The projection to use. Default is 'web-mercator'.
    levels : int, optional
        The number of levels to create. If None, the number of levels is
        determined by the number of tiles in the dataset.
    pixels_per_tile : int, optional
        Number of pixels per tile, by default 128
    other_chunks : dict
        Chunks for non-spatial dims to pass to :py:meth:`~xr.Dataset.chunk`. Default is None
    clear_attrs : bool, False
        Clear the attributes of the DataArrays within the multiscale pyramid. Default is False.

    Returns
    -------
    dt.DataTree
        The multiscale pyramid.

    """
    if not levels:
        levels = get_levels(ds)
    save_kwargs = {'levels': levels, 'pixels_per_tile': pixels_per_tile}
    attrs = {
        'multiscales': multiscales_template(
            datasets=[{'path': str(i)} for i in range(levels)],
            type='reduce',
            method='pyramid_resample',
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    # set up pyramid
    plevels = {}

    # pyramid data
    for level in range(levels):
        plevels[str(level)] = level_resample(
            ds,
            projection=projection,
            level=level,
            pixels_per_tile=pixels_per_tile,
            clear_attrs=clear_attrs,
        )

    # create the final multiscale pyramid
    plevels['/'] = xr.Dataset(attrs=attrs)
    pyramid = dt.DataTree.from_dict(plevels)

    projection_model = Projection(name=projection)

    pyramid = add_metadata_and_zarr_encoding(
        pyramid,
        levels=levels,
        pixels_per_tile=pixels_per_tile,
        other_chunks=other_chunks,
        projection=projection_model,
    )
    return pyramid
