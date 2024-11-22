import numpy as np
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from ndpyramid import (
    pyramid_reproject,
    pyramid_resample,
)
from ndpyramid.testing import verify_bounds


@pytest.mark.parametrize("resampling", ["bilinear", "nearest"])
def test_resampled_pyramid(temperature, benchmark, resampling):
    pytest.importorskip("pyresample")
    pytest.importorskip("rioxarray")
    levels = 2
    pyramid = benchmark(
        lambda: pyramid_resample(
            temperature, levels=levels, x="lon", y="lat", resampling=resampling
        )
    )
    verify_bounds(pyramid)
    assert pyramid.ds.attrs["multiscales"]
    assert len(pyramid.ds.attrs["multiscales"][0]["datasets"]) == levels
    assert pyramid.attrs["multiscales"][0]["datasets"][0]["crs"] == "EPSG:3857"
    assert pyramid["0"].attrs["multiscales"][0]["datasets"][0]["crs"] == "EPSG:3857"
    pyramid.to_zarr(MemoryStore())


@pytest.mark.xfail(reseason="Need to fix resampling of 2D data (tied to other_chunks issue)")
@pytest.mark.parametrize("method", ["bilinear", "nearest", {"air": "nearest"}])
def test_resampled_pyramid_2D(temperature, method, benchmark):
    pytest.importorskip("pyresample")
    pytest.importorskip("rioxarray")
    levels = 2
    temperature = temperature.isel(time=0).drop_vars("time")
    pyramid = benchmark(
        lambda: pyramid_resample(temperature, levels=levels, x="lon", y="lat", resampling=method)
    )
    verify_bounds(pyramid)
    assert pyramid.ds.attrs["multiscales"]
    assert len(pyramid.ds.attrs["multiscales"][0]["datasets"]) == levels
    assert pyramid.attrs["multiscales"][0]["datasets"][0]["crs"] == "EPSG:3857"
    assert pyramid["0"].attrs["multiscales"][0]["datasets"][0]["crs"] == "EPSG:3857"
    pyramid.to_zarr(MemoryStore())


def test_reprojected_pyramid_clear_attrs(dataset_3d, benchmark):
    pytest.importorskip("rioxarray")
    levels = 2
    pyramid = benchmark(
        lambda: pyramid_resample(dataset_3d, levels=levels, x="x", y="y", clear_attrs=True)
    )
    verify_bounds(pyramid)
    for _, da in pyramid["0"].ds.items():
        assert not da.attrs
    pyramid.to_zarr(MemoryStore())


@pytest.mark.xfail(reseason="Need to fix handling of other_chunks")
def test_reprojected_pyramid_other_chunks(dataset_3d, benchmark):
    pytest.importorskip("rioxarray")
    levels = 2
    pyramid = benchmark(
        lambda: pyramid_resample(dataset_3d, levels=levels, x="x", y="y", other_chunks={"time": 5})
    )
    verify_bounds(pyramid)
    pyramid.to_zarr(MemoryStore())


def test_resampled_pyramid_without_CF(dataset_3d, benchmark):
    pytest.importorskip("pyresample")
    pytest.importorskip("rioxarray")
    levels = 2
    pyramid = benchmark(lambda: pyramid_resample(dataset_3d, levels=levels, x="x", y="y"))
    verify_bounds(pyramid)
    assert pyramid.ds.attrs["multiscales"]
    assert len(pyramid.ds.attrs["multiscales"][0]["datasets"]) == levels
    assert pyramid.attrs["multiscales"][0]["datasets"][0]["crs"] == "EPSG:3857"
    assert pyramid["0"].attrs["multiscales"][0]["datasets"][0]["crs"] == "EPSG:3857"
    pyramid.to_zarr(MemoryStore())


def test_resampled_pyramid_fill(temperature, benchmark):
    """Test for https://github.com/carbonplan/ndpyramid/issues/93."""
    pytest.importorskip("pyresample")
    pytest.importorskip("rioxarray")
    pyramid = benchmark(lambda: pyramid_resample(temperature, levels=1, x="lon", y="lat"))
    assert np.isnan(pyramid["0"].air.isel(time=0, x=0, y=0).values)


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(
            "bilinear",
            marks=pytest.mark.xfail(reason="Need to investigate differences for bilinear"),
        ),
        "nearest",
    ],
)
def test_reprojected_resample_pyramid_values(dataset_3d, method, benchmark):
    pytest.importorskip("rioxarray")
    levels = 2
    reprojected = pyramid_reproject(dataset_3d, levels=levels, resampling=method)
    resampled = pyramid_resample(dataset_3d, levels=levels, x="x", y="y", resampling=method)
    xr.testing.assert_allclose(reprojected["0"].ds, resampled["0"].ds)
    xr.testing.assert_allclose(reprojected["1"].ds, resampled["1"].ds)
