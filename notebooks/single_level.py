import xarray as xr

from ndpyramid import reproject_single_level

VERSION = 2
LEVELS = 6
PIXELS_PER_TILE = 128


store = 'https://ncsa.osn.xsede.org/Pangeo/pangeo-cmems-duacs'
ds = xr.open_dataset(store, engine='zarr', chunks={})
ds = ds.isel(time=slice(0, 2))


level_ds = reproject_single_level(ds, level=1)
