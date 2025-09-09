import numpy as np
import pytest
from zarr.storage import MemoryStore

from ndpyramid import (
    pyramid_regrid,
)
from ndpyramid.regrid import generate_weights_pyramid, make_grid_ds
from ndpyramid.testing import verify_bounds


@pytest.mark.parametrize("regridder_apply_kws", [None, {"keep_attrs": False}])
def test_regridded_pyramid(temperature, regridder_apply_kws, benchmark):
    pytest.importorskip("xesmf")
    # Select subset to speed up tests
    temperature = temperature.isel(time=slice(0, 5))
    pyramid = benchmark(
        lambda: pyramid_regrid(
            temperature,
            levels=2,
            parallel_weights=False,
            regridder_apply_kws=regridder_apply_kws,
            other_chunks={"time": 2},
        )
    )
    verify_bounds(pyramid)
    assert pyramid.ds.attrs["multiscales"]
    assert pyramid.attrs["multiscales"][0]["datasets"][0]["crs"] == "EPSG:3857"
    assert pyramid["0"].attrs["multiscales"][0]["datasets"][0]["crs"] == "EPSG:3857"
    expected_attrs = (
        temperature["air"].attrs
        if not regridder_apply_kws or regridder_apply_kws.get("keep_attrs")
        else {}
    )
    assert pyramid["0"].ds.air.attrs == expected_attrs
    assert pyramid["1"].ds.air.attrs == expected_attrs
    pyramid.to_zarr(MemoryStore(), zarr_format=2)


def test_regridded_pyramid_with_weights(temperature, benchmark):
    pytest.importorskip("xesmf")
    levels = 2
    # Select subset to speed up tests
    temperature = temperature.isel(time=slice(0, 5))
    weights_pyramid = generate_weights_pyramid(temperature.isel(time=0), levels)
    pyramid = benchmark(
        lambda: pyramid_regrid(
            temperature, levels=levels, weights_pyramid=weights_pyramid, other_chunks={"time": 2}
        )
    )
    verify_bounds(pyramid)
    assert pyramid.ds.attrs["multiscales"]
    assert len(pyramid.ds.attrs["multiscales"][0]["datasets"]) == levels
    pyramid.to_zarr(MemoryStore(), zarr_format=2)


@pytest.mark.parametrize("projection", ["web-mercator", "equidistant-cylindrical"])
def test_make_grid_ds(projection, benchmark):

    grid = benchmark(lambda: make_grid_ds(0, pixels_per_tile=8, projection=projection))
    lon_vals = grid.lon_b.values
    assert np.all((lon_vals[-1, :] - lon_vals[0, :]) < 0.001)
    assert (
        grid.attrs["title"] == "Web Mercator Grid"
        if projection == "web-mercator"
        else "Equidistant Cylindrical Grid"
    )


@pytest.mark.parametrize("levels", [1, 2])
@pytest.mark.parametrize("method", ["bilinear", "conservative"])
def test_generate_weights_pyramid(temperature, levels, method, benchmark):
    pytest.importorskip("xesmf")
    weights_pyramid = benchmark(
        lambda: generate_weights_pyramid(temperature.isel(time=0), levels, method=method)
    )
    assert weights_pyramid.ds.attrs["levels"] == levels
    assert weights_pyramid.ds.attrs["regrid_method"] == method
    assert set(weights_pyramid["0"].ds.data_vars) == {"S", "col", "row"}
    assert "n_in" in weights_pyramid["0"].ds.attrs and "n_out" in weights_pyramid["0"].ds.attrs


def test_regridded_pyramid_sparse_levels(temperature):
    pytest.importorskip("xesmf")
    ds = temperature.isel(time=slice(0, 3))
    sparse_levels = [1, 3]
    pyramid = pyramid_regrid(ds, level_list=sparse_levels, parallel_weights=False)
    # DataTree keys exclude root
    assert set(pyramid.keys()) == {"1", "3"}
    datasets_meta = pyramid.ds.attrs["multiscales"][0]["datasets"]
    assert [d["level"] for d in datasets_meta] == sparse_levels
    # verify dimension sizes scale as expected
    pixels_per_tile = datasets_meta[0]["pixels_per_tile"]
    for lvl in sparse_levels:
        assert pyramid[str(lvl)].ds.dims["x"] == pixels_per_tile * 2**lvl
        assert pyramid[str(lvl)].ds.dims["y"] == pixels_per_tile * 2**lvl
