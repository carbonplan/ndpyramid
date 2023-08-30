# Usage

Ndpyramid provides a set of utilities for creating pyramids with standardized metadata.
The example below demonstrates the usage of the `pyramid_coarsen` and `pyramid_reproject`
utilities. Check out [this](https://github.com/carbonplan/ndpyramid/blob/main/notebooks/demo.ipynb)
Jupyter Notebook for a complete demonstration.

```python
import xarray as xr
import rioxarray
from ndpyramid import pyramid_coarsen, pyramid_reproject

# load a sample xarray.Dataset
ds = xr.tutorial.load_dataset('air_temperature')

# make a coarsened pyramid
pyramid = pyramid_coarsen(ds, factors=[16, 8, 4, 3, 2, 1], dims=['lat', 'lon'], boundary='trim')

# make a reprojected (EPSG:3857) pyramid
ds = ds.rio.write_crs('EPSG:4326')
pyramid = pyramid_reproject(ds, levels=2)

# write the pyramid to zarr
pyramid.to_zarr('./path/to/write')
```
