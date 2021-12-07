import numpy as np
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from ndpyramid import pyramid_coarsen, pyramid_regrid, pyramid_reproject
from ndpyramid.regrid import make_grid_ds


@pytest.fixture
def temperature():
    ds = xr.tutorial.open_dataset('air_temperature')
    ds['air'].encoding = {}
    return ds


def test_xarray_coarsened_pyramid(temperature):
    print(temperature)
    factors = [4, 2, 1]
    pyramid = pyramid_coarsen(temperature, dims=('lat', 'lon'), factors=factors, boundary='trim')
    assert pyramid.ds.attrs['multiscales']
    assert len(pyramid.ds.attrs['multiscales'][0]['datasets']) == len(factors)
    pyramid.to_zarr(MemoryStore())


def test_reprojected_pyramid(temperature):
    rioxarray = pytest.importorskip("rioxarray")  # noqa: F841
    levels = 2
    temperature = temperature.rio.write_crs('EPSG:4326')
    pyramid = pyramid_reproject(temperature, levels=2)
    assert pyramid.ds.attrs['multiscales']
    assert len(pyramid.ds.attrs['multiscales'][0]['datasets']) == levels
    pyramid.to_zarr(MemoryStore())


def test_reprojected_pyramid_dask(temperature):
    rioxarray = pytest.importorskip("rioxarray")  # noqa: F841
    levels = 2
    temperature = temperature.rio.write_crs('EPSG:4326')
    print(temperature)
    pyramid = pyramid_reproject(temperature.chunk({'time': 1}), levels=2)
    for child in pyramid.children:
        child.ds = child.ds.chunk({"x": 128, "y": 128})
    print(pyramid['0'].ds)
    assert pyramid.ds.attrs['multiscales']
    assert len(pyramid.ds.attrs['multiscales'][0]['datasets']) == levels
    pyramid.to_zarr(MemoryStore())


def test_regridded_pyramid(temperature):
    xesmf = pytest.importorskip("xesmf")  # noqa: F841
    pyramid = pyramid_regrid(temperature, levels=2)
    assert pyramid.ds.attrs['multiscales']
    pyramid.to_zarr(MemoryStore())


def test_make_grid_ds():

    grid = make_grid_ds(0, pixels_per_tile=8)
    lon_vals = grid.lon_b.values
    assert np.all((lon_vals[-1, :] + lon_vals[0, :]) < 0.001)
