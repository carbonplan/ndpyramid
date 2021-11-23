import pytest
import xarray as xr
from zarr.storage import MemoryStore

from ndpyramid import pyramid_coarsen, pyramid_regrid, pyramid_reproject


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


def test_regridded_pyramid(temperature):
    xesmf = pytest.importorskip("xesmf")  # noqa: F841
    pyramid = pyramid_regrid(temperature, levels=2)
    assert pyramid.ds.attrs['multiscales']
    pyramid.to_zarr(MemoryStore())
