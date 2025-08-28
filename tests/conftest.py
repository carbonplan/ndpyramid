import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.xr import assign_crs

# Constants
MAX_WEBMERC_LAT = 85.0511287798066  # Valid latitude domain for EPSG:3857 (Web Mercator)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _build_lon(nx: int) -> np.ndarray:
    """
    Return longitude centers spanning (-180, 180) with uniform spacing.
    Current formula yields centers from:
        (0.5 * 360/nx - 180)  ..  ( (nx-0.5) * 360/nx - 180 )
    """
    return (np.arange(nx) + 0.5) * 360.0 / nx - 180.0


def _build_lat(ny: int) -> np.ndarray:
    """
    Return latitude centers spanning (-90, 90) (exclusive of the poles)
    using the same pattern as longitude.
    """
    return (np.arange(ny) + 0.5) * 180.0 / ny - 90.0


def _maybe_clip_lat_for_web_mercator(y_full: np.ndarray, web_mercator_safe: bool) -> np.ndarray:
    """
    Optionally clip latitude centers to the valid Web Mercator extent.
    """
    if not web_mercator_safe:
        return y_full
    mask = (y_full >= -MAX_WEBMERC_LAT) & (y_full <= MAX_WEBMERC_LAT)
    return y_full[mask]


def _assert_web_mercator_safe(y: np.ndarray):
    """
    Raise a clear error if latitude extent exceeds Web Mercator domain.
    """
    if y.min() < -MAX_WEBMERC_LAT or y.max() > MAX_WEBMERC_LAT:
        raise ValueError(
            f"Latitude extent ({y.min():.6f}, {y.max():.6f}) exceeds Web Mercator limit "
            f"±{MAX_WEBMERC_LAT}. Clip first (web_mercator_safe=True) or choose another CRS."
        )


def _encode_time(ds: xr.Dataset, start_time: pd.Timestamp):
    """
    Apply CF-like time encoding referencing the dataset's first time stamp.
    """
    ds.time.encoding = {
        "units": f"days since {start_time.strftime('%Y-%m-%d')}",
        "calendar": "proleptic_gregorian",
    }
    return ds


# ---------------------------------------------------------------------------
# Base tutorial dataset fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def temperature():
    """
    Return the tutorial air temperature dataset with:
      - Time coerced to seconds resolution (avoids zarr-js int32 issues).
      - Longitude wrapped to (-180, 180).
      - CRS assigned (EPSG:4326).
      - Chunked for downstream pyramid tests.
    """
    ds = xr.tutorial.open_dataset("air_temperature")

    # Normalize time resolution
    time = ds.time.astype("datetime64[s]")
    ds = ds.assign_coords(time=time)

    # Wrap lon to (-180, 180)
    lon = ds["lon"].where(ds["lon"] < 180, ds["lon"] - 360)
    ds = ds.assign_coords(lon=lon)

    # Ensure strictly increasing longitude for tools that assume monotonic
    if not (ds["lon"].diff(dim="lon") > 0).all():
        ds = ds.reindex(lon=np.sort(ds["lon"].data))

    # Adjust lon_bounds if present
    if "lon_bounds" in ds.variables:
        lon_b = ds["lon_bounds"].where(ds["lon_bounds"] < 180, ds["lon_bounds"] - 360)
        ds = ds.assign_coords(lon_bounds=lon_b)

    # Transpose for consistent (time, lat, lon) ordering
    ds = ds.transpose("time", "lat", "lon")

    # Assign CRS and chunk
    ds = assign_crs(ds, "EPSG:4326")
    ds = ds.chunk({"time": 1000, "lat": 20, "lon": 20})
    return ds


# ---------------------------------------------------------------------------
# 4D dataset fixtures
# ---------------------------------------------------------------------------


def _make_dataset_4d(
    nb: int = 2,
    nt: int = 10,
    ny: int = 740,
    nx: int = 1440,
    start: str = "2010-01-01",
    non_dim_coords: bool = False,
    web_mercator_safe: bool = False,
    enforce_safe: bool = False,
    crs: str = "EPSG:4326",
    seed: int | None = 0,
) -> xr.Dataset:
    """
    Construct a synthetic 4D dataset (band, time, y, x).

    Parameters
    ----------
    web_mercator_safe :
        If True, clip latitude to ±85.05112878 for Web Mercator compatibility.
    enforce_safe :
        If True, raise if dataset is NOT Web Mercator safe (only checked when
        web_mercator_safe is False).
    crs :
        CRS string to assign (metadata only).
    seed :
        Random seed for reproducibility (None leaves RNG unseeded).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    time = pd.date_range(start=start, periods=nt, freq="D")

    x = _build_lon(nx)
    x_attrs = {"units": "degrees_east", "xg_name": "xgitude"}

    y_full = _build_lat(ny)
    y_full_attrs = {"units": "degrees_north", "xg_name": "yitude"}
    y = _maybe_clip_lat_for_web_mercator(y_full, web_mercator_safe)
    if enforce_safe and not web_mercator_safe:
        _assert_web_mercator_safe(y)

    ny_eff = y.size

    band = np.arange(nb)

    # Build data AFTER knowing ny_eff
    ones = np.ones((nb, nt, ny_eff, nx), dtype="float64")
    rand = rng.random((nb, nt, ny_eff, nx))

    dims = ("band", "time", "y", "x")
    coords: dict[str, tuple] = {
        "band": ("band", band),
        "time": ("time", time),
        "y": ("y", y, y_full_attrs),
        "x": ("x", x, x_attrs),
    }
    if non_dim_coords:
        coords["timestep"] = ("time", np.arange(nt))
        coords["baz"] = (("y", "x"), rng.random((ny_eff, nx)))

    ds = xr.Dataset(
        {
            "rand": (dims, rand, {"xg_name": "Beautiful Bar"}),
            "ones": (dims, ones, {"xg_name": "Fantastic Foo"}),
        },
        coords=coords,
        attrs={
            "conventions": "CF 1.6",
            "web_mercator_safe": str(web_mercator_safe),
        },
    )

    ds = _encode_time(ds, time[0])
    ds = assign_crs(ds, crs)
    return ds


@pytest.fixture()
def dataset_4d():
    """
    Default 4D dataset including polar latitudes (NOT Web Mercator safe).
    """
    return _make_dataset_4d()


@pytest.fixture()
def dataset_4d_webm():
    """
    Web Mercator safe 4D dataset (lat clipped to ±85.05112878).
    """
    return _make_dataset_4d(web_mercator_safe=True)


@pytest.fixture()
def dataset_4d_factory():
    """
    Factory fixture returning a builder function for customized 4D datasets.

    Usage in a test:
        def test_custom(dataset_4d_factory):
            ds = dataset_4d_factory(web_mercator_safe=True, nb=3)
            ...
    """

    def _factory(**kwargs):
        return _make_dataset_4d(**kwargs)

    return _factory


# ---------------------------------------------------------------------------
# 3D dataset fixtures
# ---------------------------------------------------------------------------


def _make_dataset_3d(
    nt: int = 10,
    ny: int = 740,
    nx: int = 1440,
    start: str = "2010-01-01",
    non_dim_coords: bool = False,
    web_mercator_safe: bool = False,
    enforce_safe: bool = False,
    crs: str = "EPSG:4326",
    seed: int | None = 0,
    chunks: dict | None = None,
) -> xr.Dataset:
    """
    Construct a synthetic 3D dataset (time, y, x).

    Parameters mirror _make_dataset_4d except for band dimension removal.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    time = pd.date_range(start=start, periods=nt, freq="D")

    x = _build_lon(nx)
    x_attrs = {"units": "degrees_east", "xg_name": "xgitude"}

    y_full = _build_lat(ny)
    y_attrs = {"units": "degrees_north", "xg_name": "yitude"}
    y = _maybe_clip_lat_for_web_mercator(y_full, web_mercator_safe)
    if enforce_safe and not web_mercator_safe:
        _assert_web_mercator_safe(y)

    ny_eff = y.size

    ones = np.ones((nt, ny_eff, nx), dtype="float64")
    rand = rng.random((nt, ny_eff, nx))

    dims = ("time", "y", "x")
    coords: dict[str, tuple] = {
        "time": ("time", time),
        "y": ("y", y, y_attrs),
        "x": ("x", x, x_attrs),
    }
    if non_dim_coords:
        coords["timestep"] = ("time", np.arange(nt))
        coords["baz"] = (("y", "x"), rng.random((ny_eff, nx)))

    ds = xr.Dataset(
        {
            "rand": (dims, rand, {"xg_name": "Beautiful Bar"}),
            "ones": (dims, ones, {"xg_name": "Fantastic Foo"}),
        },
        coords=coords,
        attrs={
            "conventions": "CF 1.6",
            "web_mercator_safe": str(web_mercator_safe),
        },
    )

    ds = _encode_time(ds, time[0])
    ds = assign_crs(ds, crs)

    if chunks is None:
        # Default chunking similar to original file
        ds = ds.chunk({"x": 100, "y": 100, "time": nt})
    else:
        ds = ds.chunk(chunks)

    return ds


@pytest.fixture()
def dataset_3d():
    """
    Default 3D dataset including polar latitudes (NOT Web Mercator safe).
    """
    return _make_dataset_3d()


@pytest.fixture()
def dataset_3d_webm():
    """
    Web Mercator safe 3D dataset.
    """
    return _make_dataset_3d(web_mercator_safe=True)


@pytest.fixture()
def dataset_3d_factory():
    """
    Factory fixture returning a builder for customized 3D datasets.

    Example:
        def test_reproject(dataset_3d_factory):
            ds = dataset_3d_factory(web_mercator_safe=True, chunks={'time': 5, 'y': 200, 'x': 200})
            ...
    """

    def _factory(**kwargs):
        return _make_dataset_3d(**kwargs)

    return _factory
