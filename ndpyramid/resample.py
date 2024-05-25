from __future__ import annotations  # noqa: F401

import datatree as dt
import numpy as np
import xarray as xr
from pyresample.area_config import create_area_def
from pyresample.future.resamplers.resampler import add_crs_xy_coords, update_resampled_coords
from pyresample.gradient import (
    block_bilinear_interpolator,
    gradient_resampler_indices_block,
)
from pyresample.resampler import resample_blocks
from pyresample.utils.cf import load_cf_area

from .common import Projection, ProjectionOptions
from .utils import (
    add_metadata_and_zarr_encoding,
    get_levels,
    get_version,
    multiscales_template,
)


def _create_target_area(*, dim, projection_model):
    return create_area_def(
        area_id=projection_model.name,
        projection=projection_model._crs,
        shape=(dim, dim),
        area_extent=projection_model._area_extent,
    )


def _create_source_area(da):
    ds = da.to_dataset(name='var')
    return load_cf_area(ds, variable='var')[0]


def _da_resample(da, *, dim, projection_model, pixels_per_tile, other_chunk):
    if da.encoding.get('_FillValue') is None and np.issubdtype(da.dtype, np.floating):
        da.encoding['_FillValue'] = np.nan
    target_area_def = _create_target_area(dim=dim, projection_model=projection_model)
    source_area_def = _create_source_area(da)
    indices_xy = resample_blocks(
        gradient_resampler_indices_block,
        source_area_def,
        [],
        target_area_def,
        chunk_size=(other_chunk, pixels_per_tile, pixels_per_tile),
        dtype=float,
    )
    resampled = resample_blocks(
        block_bilinear_interpolator,
        source_area_def,
        [da.data],
        target_area_def,
        dst_arrays=[indices_xy],
        chunk_size=(other_chunk, pixels_per_tile, pixels_per_tile),
        dtype=da.dtype,
    )
    resampled_da = xr.DataArray(resampled, dims=('time', 'y', 'x'))
    resampled_da = update_resampled_coords(da, resampled_da, target_area_def)
    resampled_da = add_crs_xy_coords(resampled_da, target_area_def)
    resampled_da = resampled_da.drop_vars('crs')
    resampled_da.attrs = {}
    return resampled_da


def level_resample(
    ds: xr.Dataset,
    *,
    projection: ProjectionOptions = 'web-mercator',
    level: int,
    pixels_per_tile: int = 128,
    other_chunks: dict = None,
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
    other_chunks : dict
        Chunks for non-spatial dims.
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

    dim = 2**level * pixels_per_tile
    projection_model = Projection(name=projection)
    save_kwargs = {'pixels_per_tile': pixels_per_tile}
    attrs = {
        'multiscales': multiscales_template(
            datasets=[{'path': '.', 'level': level, 'crs': projection_model._crs}],
            type='reduce',
            method='pyramid_resample',
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

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
            if other_chunks is None:
                other_chunk = list(da.sizes.values())[0]
            else:
                other_chunk = list(other_chunks.values())[0]
            ds_level[k] = _da_resample(
                da,
                dim=dim,
                projection_model=projection_model,
                pixels_per_tile=pixels_per_tile,
                other_chunk=other_chunk,
            )
    ds_level.attrs['multiscales'] = attrs['multiscales']
    return ds_level


def pyramid_resample(
    ds: xr.Dataset,
    *,
    x: str,
    y: str,
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
    y : string
        name of the variable to use as 'y' axis of the CF area definition
    x : string
        name of the variable to use as 'x' axis of the CF area definition
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
    ds = ds.rename({x: 'x', y: 'y'})

    # pyramid data
    for level in range(levels):
        plevels[str(level)] = level_resample(
            ds,
            projection=projection,
            level=level,
            pixels_per_tile=pixels_per_tile,
            other_chunks=other_chunks,
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
