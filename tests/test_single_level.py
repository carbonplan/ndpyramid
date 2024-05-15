import pytest
from zarr.storage import MemoryStore

from ndpyramid.regrid import level_regrid
from ndpyramid.reproject import level_reproject


@pytest.mark.parametrize('regridder_apply_kws', [None, {'keep_attrs': False}])
def test_level_regrid(temperature, regridder_apply_kws, benchmark):
    pytest.importorskip('xesmf')
    regrid_ds = benchmark(
        lambda: level_regrid(
            temperature, level=1, regridder_apply_kws=regridder_apply_kws, other_chunks={'time': 2}
        )
    )
    assert regrid_ds.attrs['multiscales']
    assert regrid_ds.attrs['multiscales'][0]['datasets'][0]['crs'] == 'EPSG:3857'
    expected_attrs = (
        temperature['air'].attrs
        if not regridder_apply_kws or regridder_apply_kws.get('keep_attrs')
        else {}
    )
    assert regrid_ds.air.attrs == expected_attrs
    regrid_ds.to_zarr(MemoryStore())


def test_reprojected_pyramid(temperature, benchmark):
    pytest.importorskip('rioxarray')
    temperature = temperature.rio.write_crs('EPSG:4326')
    reproject_ds = benchmark(lambda: level_reproject(temperature, level=1))
    assert reproject_ds.attrs['multiscales']
    assert len(reproject_ds.attrs['multiscales'][0]['datasets']) == 1
    assert reproject_ds.attrs['multiscales'][0]['datasets'][0]['crs'] == 'EPSG:3857'

    reproject_ds.to_zarr(MemoryStore())
