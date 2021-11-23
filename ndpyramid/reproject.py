from __future__ import annotations  # noqa: F401

from collections import defaultdict
from typing import Dict, Tuple, TypeVar

import dask
import datatree as dt
import numpy as np
import xarray as xr

from .utils import get_version, multiscales_template

ResamplingType = TypeVar('ResamplingType', str, Dict[str, str])


def _add_x_y_coords(da: xr.DataArray, shape: tuple[int], transform) -> xr.DataArray:
    '''helper function to add x/y coordinates to xr.DataArray'''

    bounds_shape = tuple(s + 1 for s in shape)

    xs = np.empty(shape)
    ys = np.empty(shape)
    for i in range(bounds_shape[0]):
        for j in range(bounds_shape[1]):
            if i < shape[0] and j < shape[1]:
                x, y = transform * [j + 0.5, i + 0.5]
                xs[i, j] = x
                ys[i, j] = y

    da = da.assign_coords(
        {'x': xr.DataArray(xs[0, :], dims=['x']), 'y': xr.DataArray(ys[:, 0], dims=['y'])}
    )

    return da


def _make_template(shape: tuple[int], dst_transform, attrs: dict) -> xr.DataArray:
    '''helper function to make a xr.DataArray template'''

    template = xr.DataArray(
        data=dask.array.empty(shape, chunks=shape), dims=('y', 'x'), attrs=attrs
    )
    template = _add_x_y_coords(template, shape, dst_transform)
    template.coords['spatial_ref'] = xr.DataArray(np.array(1.0))
    return template


def _reproject(da: xr.DataArray, shape=None, dst_transform=None, resampling='average'):
    '''helper function to reproject xr.DataArray objects'''
    from rasterio.warp import Resampling

    return da.rio.reproject(
        'EPSG:3857',
        resampling=Resampling[resampling],
        shape=shape,
        transform=dst_transform,
    )


def pyramid_reproject(
    ds,
    levels: int = None,
    pixels_per_tile=128,
    resampling: ResamplingType = 'average',
) -> dt.DataTree:
    """[summary]

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    levels : int, optional
        Number of levels in pyramid, by default None
    pixels_per_tile : int, optional
        Number of pixels to include along each axis in individual tiles, by default 128
    resampling : str or dict, optional
        Rasterio resampling method. Can be provided as a string or a per-variable
        dict, by default 'average'

    Returns
    -------
    dt.DataTree
        Multiscale data pyramid
    """
    import rioxarray  # noqa: F401
    from rasterio.transform import Affine

    # multiscales spec
    save_kwargs = {"levels": levels, "pixels_per_tile": pixels_per_tile}
    attrs = {
        "multiscales": multiscales_template(
            datasets=[{"path": str(i)} for i in range(levels)],
            type="reduce",
            method="pyramid_reproject",
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    resampling_dict: ResamplingType
    if isinstance(resampling, str):
        resampling_dict = defaultdict(lambda: resampling)
    else:
        resampling_dict = resampling

    # set up pyramid
    root = xr.Dataset(attrs=attrs)
    pyramid = dt.DataTree(data_objects={'root': root})

    for level in range(levels):
        lkey = str(level)
        dim = 2**level * pixels_per_tile

        dst_transform = Affine.translation(-20026376.39, 20048966.10) * Affine.scale(
            (20026376.39 * 2) / dim, -(20048966.10 * 2) / dim
        )

        pyramid[lkey] = xr.Dataset(attrs=ds.attrs)
        shape = (dim, dim)
        chunked_dim_sizes = ()
        for k, da in ds.items():
            template_shape = (chunked_dim_sizes) + shape  # TODO: pick up here.
            template = _make_template(template_shape, dst_transform, ds[k].attrs)
            print(resampling_dict[k])
            pyramid[lkey].ds[k] = xr.map_blocks(
                _reproject,
                da,
                kwargs=dict(
                    shape=(dim, dim), dst_transform=dst_transform, resampling=resampling_dict[k]
                ),
                template=template,
            )

    return pyramid
