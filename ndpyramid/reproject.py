from __future__ import annotations  # noqa: F401

from collections import defaultdict
from collections.abc import Sequence
from typing import cast

import numpy as np
import shapely.errors
import xarray as xr
from odc.geo import CRS as OdcCRS
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject
from pyproj import CRS as PyCRS
from pyproj import Transformer

from .common import Projection, ProjectionOptions
from .layout import BBox, PyramidLayout
from .utils import add_metadata_and_zarr_encoding, get_levels, get_version, multiscales_template


def _da_reproject(da: xr.DataArray, *, geobox: GeoBox, resampling: str):
    """Reproject a DataArray to a given GeoBox.

    Notes
    -----
    - Avoids rebuilding CRS/GeoBox per call.
    - Does not mutate the source DataArray; encodings are applied on a shallow copy.
    """
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

        return xr_reproject(da_local, geobox, resampling=resampling)

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
    # extent-aware overrides
    geobox: GeoBox | None = None,
    layout_metadata: dict | None = None,
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
    if geobox is None:
        dim = 2**level * pixels_per_tile
        dst_transform = projection_model.transform(dim=dim)
        # Build CRS/GeoBox once per level and reuse
        dst_crs_odc = OdcCRS(projection_model._crs)
        dst_geobox = GeoBox((dim, dim), dst_transform, dst_crs_odc)
    else:
        dst_geobox = geobox
    save_kwargs = {
        "level": level,
        "pixels_per_tile": pixels_per_tile,
        "projection": projection,
        "resampling": resampling,
        "extra_dim": extra_dim,
        "clear_attrs": clear_attrs,
    }

    datasets_meta = {"path": ".", "level": level, "crs": projection_model._crs}
    if layout_metadata is not None:
        # store under 'geospatial' to avoid collisions
        datasets_meta["geospatial"] = layout_metadata
    attrs = {
        "multiscales": multiscales_template(
            datasets=[datasets_meta],
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
    # extent-aware options
    crop: bool = False,
    extent: tuple[float, float, float, float] | None = None,
    extent_crs: str | None = None,
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

    crop : bool, optional
        If True, build only tiles intersecting the data footprint (extent-aware
        / cropped mode). The resulting per-level arrays are rectangular
        sub-grids aligned to the global tile matrix. Default False preserves
        legacy full-world behavior.
    extent : tuple(float, float, float, float), optional
        (xmin, ymin, xmax, ymax) of the desired crop in the *target* projection
        (unless ``extent_crs`` is implemented in a later version). If omitted
        and ``crop=True`` the bounding box is inferred from the dataset's x/y
        coordinates.
    extent_crs : str, optional
        CRS of the supplied ``extent``. If provided and different from the
        target projection CRS, it is transformed to the target CRS before
        cropping. If omitted, ``extent`` (when provided) is interpreted in the
        target projection.

    Returns
    -------
    xr.DataTree
        The multiscale pyramid.

    Notes
    -----
    When ``crop=True`` each dataset entry within
    ``pyramid.ds.attrs['multiscales'][0]['datasets']`` includes a nested
    ``geospatial`` object containing:

    - tile_offset: [tile_x0, tile_y0]
    - tile_shape: [tiles_x, tiles_y]
    - shape_px: [height_px, width_px]
    - resolution: [res_x, res_y]
    - extent_full / extent_data
    - transform: affine coefficients [a,b,c,d,e,f]
    - layout: "cropped"

    These enable clients to reconstruct global tile coordinates without
    materializing empty world pixels.

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
        "crop": crop,
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

    # extent-aware preparation
    layout: PyramidLayout | None = None
    layout_bbox: BBox | None = None
    if crop:
        projection_model = Projection(name=projection)
        world_extent = projection_model._area_extent  # full extent
        layout = PyramidLayout(
            world_extent=world_extent, pixels_per_tile=pixels_per_tile, crs=projection_model._crs
        )

        target_crs = PyCRS.from_string(projection_model._crs)

        def _infer_source_bbox() -> tuple[float, float, float, float]:
            # Try odc.geobox extent first if present (more robust than coord min/max)
            try:
                geobox = ds.odc.geobox  # type: ignore[attr-defined]
                return geobox.boundingbox.bbox
            except Exception:  # pragma: no cover - fallback path
                pass
            if {"x", "y"}.issubset(ds.coords):
                return (
                    float(ds.x.min()),
                    float(ds.y.min()),
                    float(ds.x.max()),
                    float(ds.y.max()),
                )
            raise ValueError(
                "Cannot infer extent: dataset missing 'x'/'y' coords and no odc.geobox available"
            )

        if extent is None:
            # infer from source CRS (dataset CRS) and transform to target CRS if different
            if "spatial_ref" not in ds.coords:
                raise ValueError(
                    "Dataset missing 'spatial_ref' so source CRS unknown for extent inference."
                )
            source_crs_wkt = str(ds.spatial_ref)
            source_crs = PyCRS.from_string(source_crs_wkt)
            xmin_s, ymin_s, xmax_s, ymax_s = _infer_source_bbox()
            if source_crs == target_crs:
                xmin_t, ymin_t, xmax_t, ymax_t = xmin_s, ymin_s, xmax_s, ymax_s
            else:
                transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
                # project four corners, then take min/max (safer for non-linear projections)
                xs = [xmin_s, xmin_s, xmax_s, xmax_s]
                ys = [ymin_s, ymax_s, ymin_s, ymax_s]
                tx, ty = transformer.transform(xs, ys)
                xmin_t = float(min(tx))
                xmax_t = float(max(tx))
                ymin_t = float(min(ty))
                ymax_t = float(max(ty))
            layout_bbox = BBox(xmin_t, ymin_t, xmax_t, ymax_t)
        else:
            # Interpret extent in extent_crs (if provided) else target
            if extent_crs and PyCRS.from_string(extent_crs) != target_crs:
                transformer = Transformer.from_crs(
                    PyCRS.from_string(extent_crs), target_crs, always_xy=True
                )
                xs = [extent[0], extent[2]]
                ys = [extent[1], extent[3]]
                tx, ty = transformer.transform(xs, ys)
                xmin_t, xmax_t = float(min(tx)), float(max(tx))
                ymin_t, ymax_t = float(min(ty)), float(max(ty))
                layout_bbox = BBox(xmin_t, ymin_t, xmax_t, ymax_t)
            else:
                layout_bbox = BBox(*extent)

    plevels: dict[str, xr.Dataset] = {}
    for level in level_indices:
        geobox: GeoBox | None = None
        meta: dict | None = None
        if crop and layout and layout_bbox:
            out = layout.level_from_bbox(level=level, bbox=layout_bbox)
            geobox = cast(GeoBox, out["geobox"])  # type: ignore[arg-type]
            meta = cast(dict, out["metadata"])  # type: ignore[arg-type]
        plevels[str(level)] = level_reproject(
            ds,
            projection=projection,
            level=level,
            pixels_per_tile=pixels_per_tile,
            resampling=resampling,
            extra_dim=extra_dim,
            clear_attrs=clear_attrs,
            geobox=geobox,
            layout_metadata=meta,
        )
    # create the final multiscale pyramid
    plevels["/"] = xr.Dataset(attrs=attrs)
    pyramid = xr.DataTree.from_dict(plevels)

    pyramid = add_metadata_and_zarr_encoding(
        pyramid,
        levels=level_indices,
        pixels_per_tile=pixels_per_tile,
        other_chunks=other_chunks,
        projection=projection_model,
    )
    return pyramid
