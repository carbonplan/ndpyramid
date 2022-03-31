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
    factors = [4, 2, 1]
    pyramid = pyramid_coarsen(temperature, dims=('lat', 'lon'), factors=factors, boundary='trim')
    assert pyramid.ds.attrs['multiscales']
    assert len(pyramid.ds.attrs['multiscales'][0]['datasets']) == len(factors)
    pyramid.to_zarr(MemoryStore())


def test_reprojected_pyramid(temperature):
    pytest.importorskip('rioxarray')
    levels = 2
    temperature = temperature.rio.write_crs('EPSG:4326')
    pyramid = pyramid_reproject(temperature, levels=2)
    assert pyramid.ds.attrs['multiscales']
    assert len(pyramid.ds.attrs['multiscales'][0]['datasets']) == levels
    pyramid.to_zarr(MemoryStore())


@pytest.mark.parametrize('keep_attrs', [True, False])
def test_regridded_pyramid(temperature, keep_attrs):
    pytest.importorskip('xesmf')
    pyramid = pyramid_regrid(temperature, levels=2, regridder_apply_kws={'keep_attrs': keep_attrs})
    assert pyramid.ds.attrs['multiscales']
    expected_attrs = temperature['air'].attrs if keep_attrs else {}
    assert pyramid['0'].ds.air.attrs == expected_attrs
    assert pyramid['1'].ds.air.attrs == expected_attrs
    pyramid.to_zarr(MemoryStore())


def test_make_grid_ds():

    grid = make_grid_ds(0, pixels_per_tile=8)
    lon_vals = grid.lon_b.values
    assert np.all((lon_vals[-1, :] - lon_vals[0, :]) < 0.001)
