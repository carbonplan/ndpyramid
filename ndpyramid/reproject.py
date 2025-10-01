from __future__ import annotations  # noqa: F401

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import shapely.errors
import xarray as xr
from odc.geo import CRS as OdcCRS
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject

from .common import Projection, ProjectionOptions
from .utils import add_metadata_and_zarr_encoding, get_levels, get_version, multiscales_template


def _da_reproject(
    da: xr.DataArray, *, geobox: GeoBox, resampling: str, xr_reproject_kwargs: dict | None = None
) -> xr.DataArray:
    """Reproject a DataArray to a given GeoBox.

    Notes
    -----
    - Avoids rebuilding CRS/GeoBox per call.
    - Does not mutate the source DataArray; encodings are applied on a shallow copy.
    """
    xr_reproject_kwargs = xr_reproject_kwargs or {}
    try:
        # Work on a shallow copy to avoid mutating caller's encoding/attrs
        da_local = da.copy(deep=False)

        # Ensure a sensible fill value for floating types (used by rasterio/odc-geo during warp)
        if da_local.encoding.get("_FillValue") is None and np.issubdtype(
            da_local.dtype, np.floating
        ):
            enc = dict(da_local.encoding)
            enc["_FillValue"] = np.nan
            da_local.encoding = enc

        return xr_reproject(da_local, geobox, resampling=resampling, **xr_reproject_kwargs)

    # catch the GEOSException: TopologyException error from shapely and raise a more informative error in case the user runs into
    # https://github.com/opendatacube/odc-geo/issues/147
    except shapely.errors.GEOSException as e:
        raise RuntimeError(
            "Error during reprojection. This can be caused by invalid geometries in the input data. "
            "Try cleaning the geometries or using a different resampling method. If the input data contains dask-arrays, "
            "consider using .compute() to convert them to in-memory arrays before reprojection. "
            "See https://github.com/opendatacube/odc-geo/issues/147 for more details."
        ) from e


def level_reproject(
    ds: xr.Dataset,
    *,
    projection: ProjectionOptions = "web-mercator",
    level: int,
    pixels_per_tile: int = 128,
    resampling: str | dict = "average",
    extra_dim: str | None = None,
    clear_attrs: bool = False,
    xr_reproject_kwargs: dict | None = None,
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
    resampling : str or dict
        Resampling method to use. If a dict, keys are variable names and values are odc-geo supported
        methods. A string applies to all variables.
    extra_dim : str, optional
        Deprecated/ignored. Extra dimensions are handled natively by odc-geo/xarray broadcasting.
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

    # Ensure CRS is present at the dataset level
    # raise error if not present, as this is required for reprojection
    if "spatial_ref" not in ds.coords:
        raise ValueError(
            "Source Dataset has no 'spatial_ref' coordinate. Please assign a CRS to the dataset before reprojection. You can use the 'assign_crs' function from odc.geo.xr."
        )
    projection_model = Projection(name=projection)
    dim = 2**level * pixels_per_tile
    dst_transform = projection_model.transform(dim=dim)
    # Build CRS/GeoBox once per level and reuse
    dst_crs_odc = OdcCRS(projection_model._crs)
    dst_geobox = GeoBox((dim, dim), dst_transform, dst_crs_odc)
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

    # create the data array for each level (broadcast over extra dims; no Python loop)
    ds_level = xr.Dataset(attrs=ds.attrs)
    for k, da in ds.items():
        da_reprojected = _da_reproject(
            da,
            geobox=dst_geobox,
            resampling=resampling_dict[k],
            xr_reproject_kwargs=xr_reproject_kwargs,
        )
        if clear_attrs:
            da_reprojected.attrs.clear()
        ds_level[k] = da_reprojected
    ds_level.attrs["multiscales"] = attrs["multiscales"]
    return ds_level


def pyramid_reproject(
    ds: xr.Dataset,
    *,
    projection: ProjectionOptions = "web-mercator",
    levels: int | None = None,
    level_list: Sequence[int] | None = None,
    pixels_per_tile: int = 128,
    other_chunks: dict | None = None,
    resampling: str | dict = "average",
    extra_dim: str | None = None,
    clear_attrs: bool = False,
    xr_reproject_kwargs: dict | None = None,
) -> xr.DataTree:
    """Create a multiscale pyramid of a dataset via reprojection.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to create a multiscale pyramid of.
    projection : str, optional
        The projection to use. Default is 'web-mercator'.
    levels : int, optional
        The number of (contiguous) levels to create starting at 0. Mutually exclusive with
        ``level_list``. If both ``levels`` and ``level_list`` are ``None`` then an attempt is
        made to infer the number of levels via ``get_levels`` (currently not implemented).
    level_list : Sequence[int], optional
        Explicit list of zoom levels to generate (e.g. ``[4]`` to only build Z4, or
        ``[2,4,6]`` for a sparse pyramid). Mutually exclusive with ``levels``.
    pixels_per_tile : int, optional
        Number of pixels per tile, by default 128
    other_chunks : dict
        Chunks for non-spatial dims to pass to :py:meth:`~xr.Dataset.chunk`. Default is None
    resampling : str or dict, optional
        Resampling method to use. Default is 'average'. If a dict, keys are variable names and values are resampling methods.
    extra_dim : str, optional
        Deprecated/ignored. Extra dimensions are handled natively by odc-geo/xarray broadcasting.
    clear_attrs : bool, False
        Clear the attributes of the DataArrays within the multiscale pyramid. Default is False.

    Returns
    -------
    xr.DataTree
        The multiscale pyramid.

    """
    if levels is not None and level_list is not None:
        raise ValueError("Specify only one of 'levels' or 'level_list'.")

    if level_list is not None:
        # sanitize and sort unique levels
        level_indices = sorted({int(idx) for idx in level_list})
    else:
        if not levels:
            levels = get_levels(ds)
        level_indices = list(range(int(levels)))
    projection_model = Projection(name=projection)
    save_kwargs = {
        # store the explicit list for reproducibility
        "levels": level_indices,
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
                {"path": str(i), "level": i, "crs": projection_model._crs} for i in level_indices
            ],
            type="reduce",
            method="pyramid_reproject",
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    plevels = {
        str(level): level_reproject(
            ds,
            projection=projection,
            level=level,
            pixels_per_tile=pixels_per_tile,
            resampling=resampling,
            extra_dim=extra_dim,
            clear_attrs=clear_attrs,
            xr_reproject_kwargs=xr_reproject_kwargs,
        )
        for level in level_indices
    }
    # create the final multiscale pyramid
    plevels["/"] = xr.Dataset(attrs=attrs)
    pyramid = xr.DataTree.from_dict(plevels)

    pyramid = add_metadata_and_zarr_encoding(
        pyramid,
        levels=level_indices,
        pixels_per_tile=pixels_per_tile,
        other_chunks=other_chunks,
        projection=projection_model,
        rechunk=False,  # Disable rechunking here so we can control it via xr_reproject_kwargs
    )
    return pyramid
