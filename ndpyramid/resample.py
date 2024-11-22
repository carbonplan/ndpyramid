from __future__ import annotations  # noqa: F401

import typing
import warnings
from collections import defaultdict

import numpy as np
import xarray as xr
from pyproj.crs import CRS

from .common import Projection, ProjectionOptions
from .utils import (
    add_metadata_and_zarr_encoding,
    get_levels,
    get_version,
    multiscales_template,
)

ResamplingOptions = typing.Literal["bilinear", "nearest"]


def _da_resample(
    da: xr.DataArray,
    *,
    dim: int,
    projection_model: Projection,
    pixels_per_tile: int,
    other_chunk: int,
    resampling: ResamplingOptions,
):
    try:
        from pyresample.area_config import create_area_def
        from pyresample.future.resamplers.resampler import (
            add_crs_xy_coords,
            update_resampled_coords,
        )
        from pyresample.gradient import (
            block_bilinear_interpolator,
            block_nn_interpolator,
            gradient_resampler_indices_block,
        )
        from pyresample.resampler import resample_blocks
        from pyresample.utils.cf import load_cf_area
    except ImportError as e:
        raise ImportError(
            "The use of pyramid_resample requires the packages pyresample and dask"
        ) from e
    if da.encoding.get("_FillValue") is None and np.issubdtype(da.dtype, np.floating):
        da.encoding["_FillValue"] = np.nan
    if resampling == "bilinear":
        fun = block_bilinear_interpolator
    elif resampling == "nearest":
        fun = block_nn_interpolator
    else:
        raise ValueError(f"Unrecognized interpolation method {resampling} for gradient resampling.")
    target_area_def = create_area_def(
        area_id=projection_model.name,
        projection=projection_model._crs,
        shape=(dim, dim),
        area_extent=projection_model._area_extent,
    )
    try:
        source_area_def = load_cf_area(da.to_dataset(name="var"), variable="var")[0]
    except ValueError as e:
        warnings.warn(
            f"Automatic determination of source AreaDefinition from CF conventions failed with {e}."
            " Falling back to AreaDefinition creation from coordinates."
        )
        lx = da.x[0] - (da.x[1] - da.x[0]) / 2
        rx = da.x[-1] + (da.x[-1] - da.x[-2]) / 2
        uy = da.y[0] - (da.y[1] - da.y[0]) / 2
        ly = da.y[-1] + (da.y[-1] - da.y[-2]) / 2
        source_crs = CRS.from_string(da.rio.crs.to_string())
        source_area_def = create_area_def(
            area_id=2,
            projection=source_crs,
            shape=(da.sizes["y"], da.sizes["x"]),
            area_extent=(lx.values, ly.values, rx.values, uy.values),
        )
    indices_xy = resample_blocks(
        gradient_resampler_indices_block,
        source_area_def,
        [],
        target_area_def,
        chunk_size=(other_chunk, pixels_per_tile, pixels_per_tile),
        dtype=float,
    )
    resampled = resample_blocks(
        fun,
        source_area_def,
        [da.data],
        target_area_def,
        dst_arrays=[indices_xy],
        chunk_size=(other_chunk, pixels_per_tile, pixels_per_tile),
        dtype=da.dtype,
    )
    resampled_da = xr.DataArray(resampled, dims=("time", "y", "x"))
    resampled_da = update_resampled_coords(da, resampled_da, target_area_def)
    resampled_da = add_crs_xy_coords(resampled_da, target_area_def)
    resampled_da = resampled_da.drop_vars("crs")
    resampled_da.attrs = {}
    return resampled_da


def level_resample(
    ds: xr.Dataset,
    *,
    x,
    y,
    projection: ProjectionOptions = "web-mercator",
    level: int,
    pixels_per_tile: int = 128,
    other_chunks: dict = None,
    resampling: ResamplingOptions | dict = "bilinear",
    clear_attrs: bool = False,
) -> xr.Dataset:
    """Create a level of a multiscale pyramid of a dataset via resampling.

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
    level : int
        The level of the pyramid to create.
    pixels_per_tile : int, optional
        Number of pixels per tile
    other_chunks : dict
        Chunks for non-spatial dims.
    resampling : str or dict, optional
        Pyresample resampling method to use. Default is 'bilinear'.
        If a dict, keys are variable names and values are resampling methods.
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
    save_kwargs = {"pixels_per_tile": pixels_per_tile}
    attrs = {
        "multiscales": multiscales_template(
            datasets=[{"path": ".", "level": level, "crs": projection_model._crs}],
            type="reduce",
            method="pyramid_resample",
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    # Convert resampling from string to dictionary if necessary
    if isinstance(resampling, str):
        resampling_dict: dict = defaultdict(lambda: resampling)
    else:
        resampling_dict = resampling
    # update coord naming to x & y and ensure order of dims is time, y, x
    ds = ds.rename({x: "x", y: "y"})
    # create the data array for each level
    ds_level = xr.Dataset(attrs=ds.attrs)
    for k, da in ds.items():
        if clear_attrs:
            da.attrs.clear()
        if len(da.shape) > 3:
            # if extra_dim is not specified, raise an error
            raise NotImplementedError(
                "4+ dimensional datasets are not currently supported for pyramid_resample."
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
                resampling=resampling_dict[k],
            )
    ds_level.attrs["multiscales"] = attrs["multiscales"]
    ds_level = ds_level.rio.write_crs(projection_model._crs)
    return ds_level


def pyramid_resample(
    ds: xr.Dataset,
    *,
    x: str,
    y: str,
    projection: ProjectionOptions = "web-mercator",
    levels: int = None,
    pixels_per_tile: int = 128,
    other_chunks: dict = None,
    resampling: ResamplingOptions | dict = "bilinear",
    clear_attrs: bool = False,
) -> xr.DataTree:
    """Create a multiscale pyramid of a dataset via resampling.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to create a multiscale pyramid of.
    y : string
        name of the variable to use as ``y`` axis of the CF area definition
    x : string
        name of the variable to use as ``x`` axis of the CF area definition
    projection : str, optional
        The projection to use. Default is ``web-mercator``.
    levels : int, optional
        The number of levels to create. If None, the number of levels is
        determined by the number of tiles in the dataset.
    pixels_per_tile : int, optional
        Number of pixels per tile, by default 128
    other_chunks : dict
        Chunks for non-spatial dims to pass to :py:meth:`~xr.Dataset.chunk`. Default is None
    resampling : str or dict, optional
        Pyresample resampling method to use (``bilinear`` or ``nearest``). Default is ``bilinear``.
        If a dict, keys are variable names and values are resampling methods.
    clear_attrs : bool, False
        Clear the attributes of the DataArrays within the multiscale pyramid. Default is False.

    Returns
    -------
    xr.DataTree
        The multiscale pyramid.

    Warnings
    --------
    - Pyresample expects longitude ranges between -180 - 180 degrees and latitude ranges between -90 and 90 degrees.
    - 3-D datasets are expected to have a dimension order of ``(time, y, x)``.

    ``Ndpyramid`` and ``pyresample`` do not check the validity of these assumptions to improve performance.

    """
    if not levels:
        levels = get_levels(ds)
    save_kwargs = {"levels": levels, "pixels_per_tile": pixels_per_tile}
    attrs = {
        "multiscales": multiscales_template(
            datasets=[{"path": str(i)} for i in range(levels)],
            type="reduce",
            method="pyramid_resample",
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
            x=x,
            y=y,
            projection=projection,
            level=level,
            pixels_per_tile=pixels_per_tile,
            other_chunks=other_chunks,
            resampling=resampling,
            clear_attrs=clear_attrs,
        )

    # create the final multiscale pyramid
    plevels["/"] = xr.Dataset(attrs=attrs)
    pyramid = xr.DataTree.from_dict(plevels)

    projection_model = Projection(name=projection)

    pyramid = add_metadata_and_zarr_encoding(
        pyramid,
        levels=levels,
        pixels_per_tile=pixels_per_tile,
        other_chunks=other_chunks,
        projection=projection_model,
    )
    return pyramid
