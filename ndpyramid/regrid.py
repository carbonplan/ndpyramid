import pathlib

import datatree as dt
import numpy as np
import xarray as xr

from .utils import get_version, multiscales_template


def make_grid_ds(level: int, pixels_per_tile: int = 128) -> xr.Dataset:
    """Make a dataset representing a target grid

    Parameters
    ----------
    level : int
        The zoom level to compute the grid for. Level zero is the furthest out zoom level
    pixels_per_tile : int, optional
        Number of pixels to include along each axis in individual tiles, by default 128

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
    from pyproj import Proj
    from rasterio.transform import Affine

    dim = 2 ** level * pixels_per_tile

    grid_shape = (dim, dim)
    bounds_shape = (dim + 1, dim + 1)

    transform = Affine.translation(-20026376.39, 20048966.10) * Affine.scale(
        (20026376.39 * 2) / dim, -(20048966.10 * 2) / dim
    )

    p = Proj('EPSG:3857')

    xs = np.empty(grid_shape)
    ys = np.empty(grid_shape)
    lat = np.empty(grid_shape)
    lon = np.empty(grid_shape)
    lat_b = np.zeros(bounds_shape)
    lon_b = np.zeros(bounds_shape)
    for i in range(bounds_shape[0]):
        for j in range(bounds_shape[1]):
            if i < grid_shape[0] and j < grid_shape[1]:
                x, y = transform * [j, i]
                xs[i, j] = x
                ys[i, j] = y
                lonlat = p(x, y, inverse=True)
                lon[i, j] = lonlat[0]
                lat[i, j] = lonlat[1]
            lonlat = p(*(transform * [j - 0.5, i - 0.5]), inverse=True)
            lon_b[i, j] = lonlat[0]
            lat_b[i, j] = lonlat[1]

    ds = xr.Dataset(
        {
            'x': xr.DataArray(xs[0, :], dims=['x']),
            'y': xr.DataArray(ys[:, 0], dims=['y']),
            'lat': xr.DataArray(lat, dims=['y', 'x']),
            'lon': xr.DataArray(lon, dims=['y', 'x']),
            'lat_b': xr.DataArray(lat_b, dims=['y_b', 'x_b']),
            'lon_b': xr.DataArray(lon_b, dims=['y_b', 'x_b']),
        },
        attrs=dict(title='Web Mercator Grid', Convensions='CF-1.8'),
    )

    return ds


def make_grid_pyramid(levels: int = 6) -> dt.DataTree:
    """helper function to create a grid pyramid for use with xesmf

    Parameters
    ----------
    levels : int, optional
        Number of levels in pyramid, by default 6

    Returns
    -------
    pyramid : dt.DataTree
        Multiscale grid definition
    """
    data = dt.DataTree()
    for level in range(levels):
        data[str(level)] = make_grid_ds(level).chunk(-1)
    return data

    # data.to_zarr('gs://carbonplan-scratch/grids/epsg:3857/', consolidated=True)


def pyramid_regrid(
    ds: xr.Dataset,
    target_pyramid: dt.DataTree = None,
    levels: int = None,
    weights_template: str = None,
    method: str = 'bilinear',
) -> dt.DataTree:
    """Make a pyramid using xesmf's regridders

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    target_pyramid : dt.DataTree, optional
        Target grids, if not provided, they will be generated, by default None
    levels : int, optional
        Number of levels in pyramid, by default None
    weights_template : str, optional
        Filepath to write generated weights to, e.g. `'weights_{level}'`, by default None
    method : str, optional
        Regridding method. See ``xesmf.Regridder`` for valid options, by default 'bilinear'

    Returns
    -------
    pyramid : dt.DataTree
        Multiscale data pyramid
    """
    import xesmf as xe

    if target_pyramid is None:
        if levels is not None:
            target_pyramid = make_grid_pyramid(levels)
        else:
            raise ValueError('must either provide a target_pyramid or number of levels')
    if levels is None:
        levels = len(target_pyramid.keys())  # TODO: get levels from the pyramid metadata

    # multiscales spec
    save_kwargs = locals()
    del save_kwargs['ds']
    del save_kwargs['target_pyramid']
    del save_kwargs['xe']

    attrs = {
        'multiscales': multiscales_template(
            datasets=[{'path': str(i)} for i in range(levels)],
            type='reduce',
            method='pyramid_regrid',
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    # set up pyramid
    root = xr.Dataset(attrs=attrs)
    pyramid = dt.DataTree(data_objects={"root": root})

    # pyramid data
    for level in range(levels):
        grid = target_pyramid[str(level)].ds.load()

        # get the regridder object
        if not weights_template:
            regridder = xe.Regridder(ds, grid, method)
        else:
            fn = pathlib.PosixPath(weights_template.format(level=level))
            if not fn.exists():
                regridder = xe.Regridder(ds, grid, method)
                regridder.to_netcdf(filename=fn)
            else:
                regridder = xe.Regridder(ds, grid, method, weights=fn)

        # regrid
        pyramid[str(level)] = regridder(ds)

    return pyramid
