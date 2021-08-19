import pytest
import xarray as xr
from zarr.storage import MemoryStore

from ndpyramid.core import ReprojectedPyramid, XarrayCoarsenPyramid


@pytest.fixture
def temperature():
    ds = xr.tutorial.open_dataset('air_temperature')
    return ds


def test_xarray_coarsened_pyramid(temperature):
    pyramid = XarrayCoarsenPyramid(temperature, dims=('lat', 'lon'), levels=2, boundary='trim')
    pyramid.to_zarr(MemoryStore())


def test_reprojected_pyramid(temperature):
    temperature = temperature.rio.write_crs('EPSG:4326')
    pyramid = ReprojectedPyramid(temperature, levels=2)
    pyramid.to_zarr(MemoryStore())
