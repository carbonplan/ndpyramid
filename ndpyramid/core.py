from __future__ import annotations  # noqa: F401

import typing
from collections import defaultdict

import datatree as dt
import numpy as np
import xarray as xr

from ._version import __version__
from .common import Projection
from .utils import (
    add_metadata_and_zarr_encoding,
    get_levels,
    get_version,
    multiscales_template,
    set_zarr_encoding,
)


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


# single_level branch
def reproject_single_level(
    ds: xr.Dataset,
    *,
    projection: typing.Literal['web-mercator', 'equidistant-cylindrical'] = 'web-mercator',
    level: int = None,
    pixels_per_tile: int = 128,
    other_chunks: dict = None,
    resampling: str | dict = 'average',
    extra_dim: str = None,
) -> dt.DataTree:

    import rioxarray  # noqa: F401
    from rasterio.warp import Resampling

    # multiscales spec
    save_kwargs = {'levels': level, 'pixels_per_tile': pixels_per_tile}
    attrs = {
        'multiscales': multiscales_template(
            datasets=[{'path': str(i)} for i in [level]],
            type='reduce',
            method='pyramid_reproject',
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    # Convert resampling from string to dictionary if necessary
    if isinstance(resampling, str):
        resampling_dict = defaultdict(lambda: resampling)
    else:
        resampling_dict = resampling

    projection_model = Projection(name=projection)

    # set up pyramid
    plevels = {}

    # pyramid data

    lkey = str(level)
    dim = 2**level * pixels_per_tile
    dst_transform = projection_model.transform(dim=dim)

    def reproject(da, var):
        return da.rio.reproject(
            projection_model._crs,
            resampling=Resampling[resampling_dict[var]],
            shape=(dim, dim),
            transform=dst_transform,
        )

    # create the data array for each level

    plevels[lkey] = xr.Dataset(attrs=ds.attrs)
    for k, da in ds.items():
        if len(da.shape) == 4:
            # if extra_dim is not specified, raise an error
            if extra_dim is None:
                raise ValueError("must specify 'extra_dim' to iterate over 4d data")
            da_all = []
            for index in ds[extra_dim]:
                # reproject each index of the 4th dimension
                da_reprojected = reproject(da.sel({extra_dim: index}), k)
                da_all.append(da_reprojected)
            plevels[lkey][k] = xr.concat(da_all, ds[extra_dim])
        else:
            # if the data array is not 4D, just reproject it

            plevels[lkey][k] = reproject(da, k)
    level_ds = plevels[lkey]
    level_ds.attrs = attrs

    chunks = {'x': pixels_per_tile, 'y': pixels_per_tile}

    if other_chunks is not None:
        chunks |= other_chunks
    level_ds.attrs['multiscales'][0]['metadata']['kwargs']['pixels_per_tile'] = pixels_per_tile
    if projection:
        level_ds.attrs['multiscales'][0]['datasets'][0]['crs'] = projection_model._crs
        # set dataset chunks
        level_ds = level_ds.chunk(chunks)

        # set dataset encoding

        level_ds = set_zarr_encoding(
            level_ds, codec_config={'id': 'zlib', 'level': 1}, float_dtype='float32'
        )

    # set global metadata
    level_ds.attrs.update({'title': 'multiscale data pyramid', 'version': __version__})
    return level_ds


# single_level branch
def pyramid_reproject(
    ds: xr.Dataset,
    *,
    projection: typing.Literal['web-mercator', 'equidistant-cylindrical'] = 'web-mercator',
    levels: int = None,
    pixels_per_tile: int = 128,
    other_chunks: dict = None,
    resampling: str | dict = 'average',
    extra_dim: str = None,
    clear_attrs: bool = False,
) -> dt.DataTree:
    """Create a multiscale pyramid of a dataset via reprojection.

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
    resampling : str or dict, optional
        Rasterio warp resampling method to use. Default is 'average'.
        If a dict, keys are variable names and values are warp resampling methods.
    extra_dim : str, optional
        The name of the extra dimension to iterate over. Default is None.
    clear_attrs : bool, False
        Clear the attributes of the DataArrays within the multiscale pyramid. Default is False.

    Returns
    -------
    dt.DataTree
        The multiscale pyramid.

    """

    import rioxarray  # noqa: F401
    from rasterio.warp import Resampling

    if not levels:
        levels = get_levels(ds)

    # multiscales spec
    save_kwargs = {'levels': levels, 'pixels_per_tile': pixels_per_tile}
    attrs = {
        'multiscales': multiscales_template(
            datasets=[{'path': str(i)} for i in range(levels)],
            type='reduce',
            method='pyramid_reproject',
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    # Convert resampling from string to dictionary if necessary
    if isinstance(resampling, str):
        resampling_dict: dict = defaultdict(lambda: resampling)
    else:
        resampling_dict = resampling

    projection_model = Projection(name=projection)

    # set up pyramid
    plevels = {}

    # pyramid data
    for level in range(levels):
        lkey = str(level)
        dim = 2**level * pixels_per_tile
        dst_transform = projection_model.transform(dim=dim)

        def reproject(da, var):
            da.encoding['_FillValue'] = np.nan
            da = da.rio.reproject(
                projection_model._crs,
                resampling=Resampling[resampling_dict[var]],
                shape=(dim, dim),
                transform=dst_transform,
            )
            return da

        # create the data array for each level
        plevels[lkey] = xr.Dataset(attrs=ds.attrs)
        for k, da in ds.items():
            if clear_attrs:
                da.attrs.clear()
            if len(da.shape) == 4:
                # if extra_dim is not specified, raise an error
                if extra_dim is None:
                    raise ValueError("must specify 'extra_dim' to iterate over 4d data")
                da_all = []
                for index in ds[extra_dim]:
                    # reproject each index of the 4th dimension
                    da_reprojected = reproject(da.sel({extra_dim: index}), k)
                    da_all.append(da_reprojected)
                plevels[lkey][k] = xr.concat(da_all, ds[extra_dim])
            else:
                # if the data array is not 4D, just reproject it
                plevels[lkey][k] = reproject(da, k)

    # create the final multiscale pyramid
    plevels['/'] = xr.Dataset(attrs=attrs)
    pyramid = dt.DataTree.from_dict(plevels)

    pyramid = add_metadata_and_zarr_encoding(
        pyramid,
        levels=levels,
        pixels_per_tile=pixels_per_tile,
        other_chunks=other_chunks,
        projection=projection_model,
    )
    return pyramid
