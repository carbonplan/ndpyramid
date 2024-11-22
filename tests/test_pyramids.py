import numpy as np
import pytest
from zarr.storage import MemoryStore

from ndpyramid import (
    pyramid_coarsen,
    pyramid_create,
    pyramid_reproject,
)
from ndpyramid.testing import verify_bounds


def test_xarray_coarsened_pyramid(temperature, benchmark):
    factors = [4, 2, 1]
    pyramid = benchmark(
        lambda: pyramid_coarsen(temperature, dims=("lat", "lon"), factors=factors, boundary="trim")
    )
    assert pyramid.ds.attrs["multiscales"]
    assert len(pyramid.ds.attrs["multiscales"][0]["datasets"]) == len(factors)
    assert pyramid.ds.attrs["multiscales"][0]["metadata"]["method"] == "pyramid_coarsen"
    assert pyramid.ds.attrs["multiscales"][0]["type"] == "reduce"
    pyramid.to_zarr(MemoryStore())


@pytest.mark.parametrize("method_label", [None, "sel_coarsen"])
def test_xarray_custom_coarsened_pyramid(temperature, benchmark, method_label):
    def sel_coarsen(ds, factor, dims, **kwargs):
        return ds.sel(**{dim: slice(None, None, factor) for dim in dims})

    factors = [4, 2, 1]
    pyramid = benchmark(
        lambda: pyramid_create(
            temperature,
            dims=("lat", "lon"),
            factors=factors,
            boundary="trim",
            func=sel_coarsen,
            method_label=method_label,
            type_label="pick",
        )
    )
    assert pyramid.ds.attrs["multiscales"]
    assert len(pyramid.ds.attrs["multiscales"][0]["datasets"]) == len(factors)
    assert pyramid.ds.attrs["multiscales"][0]["metadata"]["method"] == "sel_coarsen"
    assert pyramid.ds.attrs["multiscales"][0]["type"] == "pick"
    pyramid.to_zarr(MemoryStore())


def test_reprojected_pyramid(temperature, benchmark):
    pytest.importorskip("rioxarray")
    levels = 2
    pyramid = benchmark(lambda: pyramid_reproject(temperature, levels=levels))
    verify_bounds(pyramid)
    assert pyramid.ds.attrs["multiscales"]
    assert len(pyramid.ds.attrs["multiscales"][0]["datasets"]) == levels
    assert pyramid.attrs["multiscales"][0]["datasets"][0]["crs"] == "EPSG:3857"
    assert pyramid["0"].attrs["multiscales"][0]["datasets"][0]["crs"] == "EPSG:3857"
    pyramid.to_zarr(MemoryStore())


def test_reprojected_pyramid_resampling_dict(dataset_3d, benchmark):
    pytest.importorskip("rioxarray")
    levels = 2
    pyramid = benchmark(
        lambda: pyramid_reproject(
            dataset_3d, levels=levels, resampling={"ones": "bilinear", "rand": "nearest"}
        )
    )
    verify_bounds(pyramid)
    assert pyramid.attrs["multiscales"][0]["metadata"]["kwargs"]["resampling"] == {
        "ones": "bilinear",
        "rand": "nearest",
    }
    pyramid.to_zarr(MemoryStore())


def test_reprojected_pyramid_clear_attrs(dataset_3d, benchmark):
    pytest.importorskip("rioxarray")
    levels = 2
    pyramid = benchmark(lambda: pyramid_reproject(dataset_3d, levels=levels, clear_attrs=True))
    verify_bounds(pyramid)
    for _, da in pyramid["0"].ds.items():
        assert not da.attrs
    pyramid.to_zarr(MemoryStore())


def test_reprojected_pyramid_4d(dataset_4d, benchmark):
    pytest.importorskip("rioxarray")
    levels = 2
    with pytest.raises(Exception):
        pyramid = pyramid_reproject(dataset_4d, levels=levels)
    pyramid = benchmark(lambda: pyramid_reproject(dataset_4d, levels=levels, extra_dim="band"))
    verify_bounds(pyramid)
    assert pyramid.ds.attrs["multiscales"]
    assert len(pyramid.ds.attrs["multiscales"][0]["datasets"]) == levels
    assert pyramid.attrs["multiscales"][0]["datasets"][0]["crs"] == "EPSG:3857"
    assert pyramid["0"].attrs["multiscales"][0]["datasets"][0]["crs"] == "EPSG:3857"
    pyramid.to_zarr(MemoryStore())


def test_reprojected_pyramid_fill(temperature, benchmark):
    """Test for https://github.com/carbonplan/ndpyramid/issues/93."""
    pytest.importorskip("rioxarray")
    pyramid = benchmark(lambda: pyramid_reproject(temperature, levels=1))
    assert np.isnan(pyramid["0"].air.isel(time=0, x=0, y=0).values)
