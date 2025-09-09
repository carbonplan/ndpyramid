import pytest

from ndpyramid import pyramid_reproject, pyramid_resample


@pytest.mark.parametrize("levels_arg", [[4], [2, 4], [1, 3, 5]])
def test_reproject_sparse_levels(temperature, levels_arg):
    ds = temperature.isel(time=slice(0, 1))  # keep small
    pyr = pyramid_reproject(ds, level_list=levels_arg)
    requested = sorted({int(level) for level in levels_arg})
    # DataTree keys exclude root; ensure only requested levels present
    assert set(pyr.keys()) == {*map(str, requested)}
    meta_levels = [d["level"] for d in pyr.ds.attrs["multiscales"][0]["datasets"]]
    assert meta_levels == requested
    # ensure dataset dimensions match expected sizes (pixels_per_tile * 2**level)
    pxt = pyr.ds.attrs["multiscales"][0]["datasets"][0]["pixels_per_tile"]
    for level in requested:
        assert pyr[str(level)].ds.dims["x"] == pxt * 2**level
        assert pyr[str(level)].ds.dims["y"] == pxt * 2**level


@pytest.mark.parametrize("levels_arg", [[3], [0, 2], [2, 3, 4]])
def test_resample_sparse_levels(temperature, levels_arg):
    ds = temperature.isel(time=slice(0, 1))  # small
    # rename coordinates to lon/lat expected by tests
    pyr = pyramid_resample(ds, x="lon", y="lat", level_list=levels_arg)
    requested = sorted({int(level) for level in levels_arg})
    assert set(pyr.keys()) == {*map(str, requested)}
    meta_paths = [d["path"] for d in pyr.ds.attrs["multiscales"][0]["datasets"]]
    assert meta_paths == [str(level) for level in requested]
