import pytest
import xarray as xr
from zarr.storage import MemoryStore

from ndpyramid.core import XarrayCoarsenPyramid


@pytest.fixture
def temperature():
    ds = xr.tutorial.open_dataset('air_temperature')
    return ds


def test_xarray_coarsened_pyramid(temperature):
    pyramid = XarrayCoarsenPyramid(temperature, dims=('lat', 'lon'), levels=2, boundary='trim')
    pyramid.to_zarr(MemoryStore())
