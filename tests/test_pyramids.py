import pytest
import xarray as xr
from zarr.storage import MemoryStore

from ndpyramid import pyramid_coarsen, pyramid_reproject


@pytest.fixture
def temperature():
    ds = xr.tutorial.open_dataset('air_temperature')
    return ds


def test_xarray_coarsened_pyramid(temperature):
    print(temperature)
    pyramid = pyramid_coarsen(temperature, dims=('lat', 'lon'), factors=[4, 2, 1], boundary='trim')
    assert pyramid.ds.attrs['multiscales']
    pyramid.to_zarr(MemoryStore())


def test_reprojected_pyramid(temperature):
    rioxarray = pytest.importorskip("rioxarray")  # noqa: F841
    temperature = temperature.rio.write_crs('EPSG:4326')
    pyramid = pyramid_reproject(temperature, levels=2)
    assert pyramid.ds.attrs['multiscales']
    pyramid.to_zarr(MemoryStore())
