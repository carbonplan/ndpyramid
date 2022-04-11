import numpy as np
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from ndpyramid import pyramid_coarsen, pyramid_regrid, pyramid_reproject
from ndpyramid.regrid import generate_weights_pyramid, make_grid_ds


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


@pytest.mark.parametrize('regridder_apply_kws', [None, {'keep_attrs': False}])
def test_regridded_pyramid(temperature, regridder_apply_kws):
    pytest.importorskip('xesmf')
    pyramid = pyramid_regrid(
        temperature, levels=2, regridder_apply_kws=regridder_apply_kws, other_chunks={'time': 2}
    )
    assert pyramid.ds.attrs['multiscales']
    expected_attrs = (
        temperature['air'].attrs
        if not regridder_apply_kws or regridder_apply_kws.get('keep_attrs')
        else {}
    )
    assert pyramid['0'].ds.air.attrs == expected_attrs
    assert pyramid['1'].ds.air.attrs == expected_attrs
    pyramid.to_zarr(MemoryStore())


def test_regridded_pyramid_with_weights(temperature):
    pytest.importorskip('xesmf')
    levels = 2
    weights_pyramid = generate_weights_pyramid(temperature.isel(time=0), levels)
    pyramid = pyramid_regrid(
        temperature, levels=levels, weights_pyramid=weights_pyramid, other_chunks={'time': 2}
    )
    assert pyramid.ds.attrs['multiscales']
    assert len(pyramid.ds.attrs['multiscales'][0]['datasets']) == levels
    pyramid.to_zarr(MemoryStore())


def test_make_grid_ds():

    grid = make_grid_ds(0, pixels_per_tile=8)
    lon_vals = grid.lon_b.values
    assert np.all((lon_vals[-1, :] - lon_vals[0, :]) < 0.001)


@pytest.mark.parametrize('levels', [1, 2])
@pytest.mark.parametrize('method', ['bilinear', 'conservative'])
def test_generate_weights_pyramid(temperature, levels, method):
    weights_pyramid = generate_weights_pyramid(temperature.isel(time=0), levels, method=method)
    assert weights_pyramid.ds.attrs['levels'] == levels
    assert weights_pyramid.ds.attrs['regrid_method'] == method
    assert set(weights_pyramid['0'].ds.data_vars) == {'S', 'col', 'row'}
    assert 'n_in' in weights_pyramid['0'].ds.attrs and 'n_out' in weights_pyramid['0'].ds.attrs
