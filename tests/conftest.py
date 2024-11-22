import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa
import xarray as xr


@pytest.fixture
def temperature():
    ds = xr.tutorial.open_dataset("air_temperature")
    # Update time coordinate to avoid errors when casting to int32 for zarr-js
    time = ds.time.astype("datetime64[s]")
    ds = ds.assign_coords(time=time)
    # Update lon coordinates to -180 to 180 for compatibility with pyresample
    # Logic copied from carbonplan/cmip6-downscaling
    lon = ds["lon"].where(ds["lon"] < 180, ds["lon"] - 360)
    ds = ds.assign_coords(lon=lon)

    if not (ds["lon"].diff(dim="lon") > 0).all():
        ds = ds.reindex(lon=np.sort(ds["lon"].data))

    if "lon_bounds" in ds.variables:
        lon_b = ds["lon_bounds"].where(ds["lon_bounds"] < 180, ds["lon_bounds"] - 360)
        ds = ds.assign_coords(lon_bounds=lon_b)
    # Transpose coordinates for compatibility with pyresample
    ds = ds.transpose("time", "lat", "lon")
    # Write crs for pyramid_resampled and pyramid_reproject
    ds = ds.rio.write_crs("EPSG:4326")
    # Chunk the dataset for testing `pyramid_resample`
    ds = ds.chunk({"time": 1000, "lat": 20, "lon": 20})
    return ds


@pytest.fixture()
def dataset_4d(non_dim_coords=False, start="2010-01-01"):
    """Return a synthetic random Xarray dataset.
    Modified from https://github.com/pangeo-forge/pangeo-forge-recipes/blob/fbaaf31b6f278418bd9ba6750ffcdb9874409196/tests/data_generation.py#L6-L45
    """
    nb, nt, ny, nx = 2, 10, 740, 1440

    time = pd.date_range(start=start, periods=nt, freq="D")
    x = (np.arange(nx) + 0.5) * 360 / nx - 180
    x_attrs = {"units": "degrees_east", "xg_name": "xgitude"}
    y = (np.arange(ny) + 0.5) * 180 / ny - 90
    y_attrs = {"units": "degrees_north", "xg_name": "yitude"}
    foo = np.ones((nb, nt, ny, nx))
    foo_attrs = {"xg_name": "Fantastic Foo"}
    bar = np.random.rand(nb, nt, ny, nx)
    bar_attrs = {"xg_name": "Beautiful Bar"}
    band = np.arange(nb)
    dims = ("band", "time", "y", "x")
    coords = {
        "band": ("band", band),
        "time": ("time", time),
        "y": ("y", y, y_attrs),
        "x": ("x", x, x_attrs),
    }
    if non_dim_coords:
        coords["timestep"] = ("time", np.arange(nt))
        coords["baz"] = (("y", "x"), np.random.rand(ny, nx))

    ds = xr.Dataset(
        {"rand": (dims, bar, bar_attrs), "ones": (dims, foo, foo_attrs)},
        coords=coords,
        attrs={"conventions": "CF 1.6"},
    )

    # Add time coord encoding
    # Remove "%H:%M:%s" as it will be dropped when time is 0:0:0
    ds.time.encoding = {
        "units": f"days since {time[0].strftime('%Y-%m-%d')}",
        "calendar": "proleptic_gregorian",
    }
    ds = ds.rio.write_crs("EPSG:4326")

    return ds


@pytest.fixture()
def dataset_3d(non_dim_coords=False, start="2010-01-01"):
    """Return a synthetic random Xarray dataset.
    Modified from https://github.com/pangeo-forge/pangeo-forge-recipes/blob/fbaaf31b6f278418bd9ba6750ffcdb9874409196/tests/data_generation.py#L6-L45
    """
    nt, ny, nx = 10, 740, 1440

    time = pd.date_range(start=start, periods=nt, freq="D")
    x = (np.arange(nx) + 0.5) * 360 / nx - 180
    x_attrs = {"units": "degrees_east", "xg_name": "xgitude"}
    y = (np.arange(ny) + 0.5) * 180 / ny - 90
    y_attrs = {"units": "degrees_north", "xg_name": "yitude"}
    foo = np.ones((nt, ny, nx))
    foo_attrs = {"xg_name": "Fantastic Foo"}
    bar = np.random.rand(nt, ny, nx)
    bar_attrs = {"xg_name": "Beautiful Bar"}
    dims = ("time", "y", "x")
    coords = {
        "time": ("time", time),
        "y": ("y", y, y_attrs),
        "x": ("x", x, x_attrs),
    }
    if non_dim_coords:
        coords["timestep"] = ("time", np.arange(nt))
        coords["baz"] = (("y", "x"), np.random.rand(ny, nx))

    ds = xr.Dataset(
        {"rand": (dims, bar, bar_attrs), "ones": (dims, foo, foo_attrs)},
        coords=coords,
        attrs={"conventions": "CF 1.6"},
    )

    # Add time coord encoding
    # Remove "%H:%M:%s" as it will be dropped when time is 0:0:0
    ds.time.encoding = {
        "units": f"days since {time[0].strftime('%Y-%m-%d')}",
        "calendar": "proleptic_gregorian",
    }
    ds = ds.rio.write_crs("EPSG:4326")
    ds = ds.chunk({"x": 100, "y": 100, "time": 10})

    return ds
