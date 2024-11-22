from __future__ import annotations  # noqa: F401

from collections import defaultdict

import numpy as np
import xarray as xr
from rasterio.warp import Resampling

from .common import Projection, ProjectionOptions
from .utils import (
    add_metadata_and_zarr_encoding,
    get_levels,
    get_version,
    multiscales_template,
)


def _da_reproject(da: xr.DataArray, *, dim: int, crs: str, resampling: str, transform):
    if da.encoding.get("_FillValue") is None and np.issubdtype(da.dtype, np.floating):
        da.encoding["_FillValue"] = np.nan
    return da.rio.reproject(
        crs,
        resampling=resampling,
        shape=(dim, dim),
        transform=transform,
    )


def level_reproject(
    ds: xr.Dataset,
    *,
    projection: ProjectionOptions = "web-mercator",
    level: int,
    pixels_per_tile: int = 128,
    resampling: str | dict = "average",
    extra_dim: str = None,
    clear_attrs: bool = False,
) -> xr.Dataset:
    """Create a level of a multiscale pyramid of a dataset via reprojection.

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
    resampling : dict
        Rasterio warp resampling method to use. Keys are variable names and values are warp resampling methods.
    extra_dim : str, optional
        The name of the extra dimension to iterate over. Default is None.
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
    projection_model = Projection(name=projection)
    dim = 2**level * pixels_per_tile
    dst_transform = projection_model.transform(dim=dim)
    save_kwargs = {
        "level": level,
        "pixels_per_tile": pixels_per_tile,
        "projection": projection,
        "resampling": resampling,
        "extra_dim": extra_dim,
        "clear_attrs": clear_attrs,
    }

    attrs = {
        "multiscales": multiscales_template(
            datasets=[{"path": ".", "level": level, "crs": projection_model._crs}],
            type="reduce",
            method="pyramid_reproject",
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    # Convert resampling from string to dictionary if necessary
    if isinstance(resampling, str):
        resampling_dict: dict = defaultdict(lambda: resampling)
    else:
        resampling_dict = resampling

    # create the data array for each level
    ds_level = xr.Dataset(attrs=ds.attrs)
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
                da_reprojected = _da_reproject(
                    da.sel({extra_dim: index}),
                    dim=dim,
                    crs=projection_model._crs,
                    resampling=Resampling[resampling_dict[k]],
                    transform=dst_transform,
                )
                da_all.append(da_reprojected)
            ds_level[k] = xr.concat(da_all, ds[extra_dim])
        else:
            # if the data array is not 4D, just reproject it
            ds_level[k] = _da_reproject(
                da,
                dim=dim,
                crs=projection_model._crs,
                resampling=Resampling[resampling_dict[k]],
                transform=dst_transform,
            )
    ds_level.attrs["multiscales"] = attrs["multiscales"]
    return ds_level


def pyramid_reproject(
    ds: xr.Dataset,
    *,
    projection: ProjectionOptions = "web-mercator",
    levels: int = None,
    pixels_per_tile: int = 128,
    other_chunks: dict = None,
    resampling: str | dict = "average",
    extra_dim: str = None,
    clear_attrs: bool = False,
) -> xr.DataTree:
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
    xr.DataTree
        The multiscale pyramid.

    """
    if not levels:
        levels = get_levels(ds)
    projection_model = Projection(name=projection)
    save_kwargs = {
        "levels": levels,
        "pixels_per_tile": pixels_per_tile,
        "projection": projection,
        "other_chunks": other_chunks,
        "resampling": resampling,
        "extra_dim": extra_dim,
        "clear_attrs": clear_attrs,
    }
    attrs = {
        "multiscales": multiscales_template(
            datasets=[
                {"path": str(i), "level": i, "crs": projection_model._crs} for i in range(levels)
            ],
            type="reduce",
            method="pyramid_reproject",
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    # set up pyramid
    plevels = {}

    # pyramid data
    for level in range(levels):
        plevels[str(level)] = level_reproject(
            ds,
            projection=projection,
            level=level,
            pixels_per_tile=pixels_per_tile,
            resampling=resampling,
            extra_dim=extra_dim,
            clear_attrs=clear_attrs,
        )

    # create the final multiscale pyramid
    plevels["/"] = xr.Dataset(attrs=attrs)
    pyramid = xr.DataTree.from_dict(plevels)

    pyramid = add_metadata_and_zarr_encoding(
        pyramid,
        levels=levels,
        pixels_per_tile=pixels_per_tile,
        other_chunks=other_chunks,
        projection=projection_model,
    )
    return pyramid
