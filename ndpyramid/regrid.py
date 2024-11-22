from __future__ import annotations  # noqa: F401

import itertools
import typing

import numpy as np
import xarray as xr

from .common import Projection
from .utils import add_metadata_and_zarr_encoding, get_version, multiscales_template


def xesmf_weights_to_xarray(regridder) -> xr.Dataset:
    w = regridder.weights.data
    dim = "n_s"
    ds = xr.Dataset(
        {
            "S": (dim, w.data),
            "col": (dim, w.coords[1, :] + 1),
            "row": (dim, w.coords[0, :] + 1),
        }
    )
    ds.attrs = {"n_in": regridder.n_in, "n_out": regridder.n_out}
    return ds


def _reconstruct_xesmf_weights(ds_w):
    """Reconstruct weights into format that xESMF understands"""
    import sparse
    import xarray as xr

    col = ds_w["col"].values - 1
    row = ds_w["row"].values - 1
    s = ds_w["S"].values
    n_out, n_in = ds_w.attrs["n_out"], ds_w.attrs["n_in"]
    crds = np.stack([row, col])
    return xr.DataArray(
        sparse.COO(crds, s, (n_out, n_in)), dims=("out_dim", "in_dim"), name="weights"
    )


def make_grid_ds(
    level: int,
    pixels_per_tile: int = 128,
    projection: typing.Literal["web-mercator", "equidistant-cylindrical"] = "web-mercator",
) -> xr.Dataset:
    """Make a dataset representing a target grid

    Parameters
    ----------
    level : int
        The zoom level to compute the grid for. Level zero is the furthest out zoom level
    pixels_per_tile : int, optional
        Number of pixels to include along each axis in individual tiles, by default 128
    projection : str, optional
        The projection to use for the grid, by default 'equidistant-cylindrical'

    Returns
    -------
    xr.Dataset
        Target grid dataset with the following variables:
        - "x": X coordinate in Web Mercator projection (grid cell center)
        - "y": Y coordinate in Web Mercator projection (grid cell center)
        - "lat": latitude coordinate (grid cell center)
        - "lon": longitude coordinate (grid cell center)
        - "lat_b": latitude bounds for grid cell
        - "lon_b": longitude bounds for grid cell

    """
    projection_model = Projection(name=projection)

    dim = (2**level) * pixels_per_tile

    transform = projection_model.transform(dim=dim)

    if projection_model.name == "equidistant-cylindrical":
        title = "Equidistant Cylindrical Grid"

    elif projection_model.name == "web-mercator":
        title = "Web Mercator Grid"

    p = projection_model._proj

    grid_shape = (dim, dim)
    bounds_shape = (dim + 1, dim + 1)

    xs = np.empty(grid_shape)
    ys = np.empty(grid_shape)
    lat = np.empty(grid_shape)
    lon = np.empty(grid_shape)
    lat_b = np.zeros(bounds_shape)
    lon_b = np.zeros(bounds_shape)

    # calc grid cell center coordinates
    ii, jj = np.meshgrid(np.arange(dim) + 0.5, np.arange(dim) + 0.5)
    for i, j in itertools.product(range(grid_shape[0]), range(grid_shape[1])):
        locs = [ii[i, j], jj[i, j]]
        xs[i, j], ys[i, j] = transform * locs
        lon[i, j], lat[i, j] = p(xs[i, j], ys[i, j], inverse=True)

    # calc grid cell bounds
    iib, jjb = np.meshgrid(np.arange(dim + 1), np.arange(dim + 1))
    for i, j in itertools.product(range(bounds_shape[0]), range(bounds_shape[1])):
        locs = [iib[i, j], jjb[i, j]]
        x, y = transform * locs
        lon_b[i, j], lat_b[i, j] = p(x, y, inverse=True)

    return xr.Dataset(
        {
            "x": xr.DataArray(xs[0, :], dims=["x"]),
            "y": xr.DataArray(ys[:, 0], dims=["y"]),
            "lat": xr.DataArray(lat, dims=["y", "x"]),
            "lon": xr.DataArray(lon, dims=["y", "x"]),
            "lat_b": xr.DataArray(lat_b, dims=["y_b", "x_b"]),
            "lon_b": xr.DataArray(lon_b, dims=["y_b", "x_b"]),
        },
        attrs=dict(title=title, Conventions="CF-1.8"),
    )


def make_grid_pyramid(
    levels: int = 6,
    projection: typing.Literal["web-mercator", "equidistant-cylindrical"] = "web-mercator",
    pixels_per_tile: int = 128,
) -> xr.DataTree:
    """Helper function to create a grid pyramid for use with xesmf

    Parameters
    ----------
    levels : int, optional
        Number of levels in pyramid, by default 6

    Returns
    -------
    pyramid : xr.DataTree
        Multiscale grid definition

    """
    plevels = {
        str(level): make_grid_ds(
            level, projection=projection, pixels_per_tile=pixels_per_tile
        ).chunk(-1)
        for level in range(levels)
    }
    return xr.DataTree.from_dict(plevels)


def generate_weights_pyramid(
    ds_in: xr.Dataset,
    levels: int,
    method: str = "bilinear",
    regridder_kws: dict = None,
    projection: typing.Literal["web-mercator", "equidistant-cylindrical"] = "web-mercator",
) -> xr.DataTree:
    """Helper function to generate weights for a multiscale regridder

    Parameters
    ----------
    ds_in : xr.Dataset
        Input dataset to regrid
    levels : int
        Number of levels in the pyramid
    method : str, optional
        Regridding method. See :py:class:`~xesmf.Regridder` for valid options, by default 'bilinear'
    regridder_kws : dict
        Keyword arguments to pass to :py:class:`~xesmf.Regridder`. Default is `{'periodic': True}`
    projection : str, optional
        The projection to use for the grid, by default 'web-mercator'

    Returns
    -------
    weights : xr.DataTree
        Multiscale weights

    """
    import xesmf as xe

    regridder_kws = {} if regridder_kws is None else regridder_kws
    regridder_kws = {"periodic": True, **regridder_kws}

    plevels = {}
    for level in range(levels):
        ds_out = make_grid_ds(level=level, projection=projection)
        regridder = xe.Regridder(ds_in, ds_out, method, **regridder_kws)
        ds = xesmf_weights_to_xarray(regridder)

        plevels[str(level)] = ds

    root = xr.Dataset(attrs={"levels": levels, "regrid_method": method})
    plevels["/"] = root
    return xr.DataTree.from_dict(plevels)


def pyramid_regrid(
    ds: xr.Dataset,
    projection: typing.Literal["web-mercator", "equidistant-cylindrical"] = "web-mercator",
    target_pyramid: xr.DataTree = None,
    levels: int = None,
    parallel_weights: bool = True,
    weights_pyramid: xr.DataTree = None,
    method: str = "bilinear",
    regridder_kws: dict = None,
    regridder_apply_kws: dict = None,
    other_chunks: dict = None,
    pixels_per_tile: int = 128,
) -> xr.DataTree:
    """Make a pyramid using xesmf's regridders

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    projection : str, optional
        Projection to use for the grid, by default 'web-mercator'
    target_pyramid : xr.DataTree, optional
        Target grids, if not provided, they will be generated, by default None
    levels : int, optional
        Number of levels in pyramid, by default None
    weights_pyramid : xr.DataTree, optional
       pyramid containing pregenerated weights
    parallel_weights : Bool
        Use dask to generate parallel weights
    method : str, optional
        Regridding method. See :py:class:`~xesmf.Regridder` for valid options, by default 'bilinear'
    regridder_kws : dict
        Keyword arguments to pass to regridder. Default is `{'periodic': True}`
    regridder_apply_kws : dict
        Keyword arguments such as `keep_attrs`, `skipna`, `na_thres`
        to pass to :py:meth:`~xesmf.Regridder.__call__`. Default is None
    other_chunks : dict
        Chunks for non-spatial dims to pass to :py:meth:`~xr.Dataset.chunk`. Default is None
    pixels_per_tile : int, optional
        Number of pixels per tile, by default 128

    Returns
    -------
    pyramid : xr.DataTree
        Multiscale data pyramid

    """
    import xesmf as xe

    if target_pyramid is None:
        if levels is not None:
            target_pyramid = make_grid_pyramid(
                levels, projection=projection, pixels_per_tile=pixels_per_tile
            )
        else:
            raise ValueError("must either provide a target_pyramid or number of levels")
    if levels is None:
        levels = len(target_pyramid.keys())  # TODO: get levels from the pyramid metadata

    regridder_kws = {} if regridder_kws is None else regridder_kws
    regridder_kws = {"periodic": True, **regridder_kws}

    # multiscales spec
    projection_model = Projection(name=projection)
    save_kwargs = {
        "levels": levels,
        "pixels_per_tile": pixels_per_tile,
        "projection": projection,
        "other_chunks": other_chunks,
        "method": method,
        "regridder_kws": regridder_kws,
        "regridder_apply_kws": regridder_apply_kws,
    }

    attrs = {
        "multiscales": multiscales_template(
            datasets=[
                {"path": str(i), "level": i, "crs": projection_model._crs} for i in range(levels)
            ],
            type="reduce",
            method="pyramid_regrid",
            version=get_version(),
            kwargs=save_kwargs,
        )
    }
    save_kwargs.pop("levels")
    save_kwargs.pop("other_chunks")

    # set up pyramid

    plevels = {}

    # pyramid data
    for level in range(levels):
        grid = target_pyramid[str(level)].ds.load()
        # get the regridder object
        if weights_pyramid is None:
            regridder = xe.Regridder(ds, grid, method, parallel=parallel_weights, **regridder_kws)
        else:
            # Reconstruct weights into format that xESMF understands
            # this is a hack that assumes the weights were generated by
            # the `generate_weights_pyramid` function

            ds_w = weights_pyramid[str(level)].ds
            weights = _reconstruct_xesmf_weights(ds_w)
            regridder = xe.Regridder(
                ds, grid, method, reuse_weights=True, weights=weights, **regridder_kws
            )
        # regrid
        if regridder_apply_kws is None:
            regridder_apply_kws = {}
        regridder_apply_kws = {**{"keep_attrs": True}, **regridder_apply_kws}
        plevels[str(level)] = regridder(ds, **regridder_apply_kws)
        level_attrs = {
            "multiscales": multiscales_template(
                datasets=[{"path": ".", "level": level, "crs": projection_model._crs}],
                type="reduce",
                method="pyramid_regrid",
                version=get_version(),
                kwargs=save_kwargs,
            )
        }
        plevels[str(level)].attrs["multiscales"] = level_attrs["multiscales"]

    root = xr.Dataset(attrs=attrs)
    plevels["/"] = root
    pyramid = xr.DataTree.from_dict(plevels)

    pyramid = add_metadata_and_zarr_encoding(
        pyramid,
        levels=levels,
        other_chunks=other_chunks,
        pixels_per_tile=pixels_per_tile,
        projection=Projection(name=projection),
    )

    return pyramid
