import pytest
import xarray as xr


@pytest.fixture
def temperature():
    ds = xr.tutorial.open_dataset('air_temperature')
    ds['air'].encoding = {}
    return ds
