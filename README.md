<img
  src='https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png'
  height='48'
/>

# ndpyramid

A small utility for generating ND array pyramids using Xarray and Zarr.

[![CI](https://github.com/carbonplan/ndpyramid/actions/workflows/main.yaml/badge.svg)](https://github.com/carbonplan/ndpyramid/actions/workflows/main.yaml)
![MIT License](https://badgen.net/badge/license/MIT/blue)

# installation

This package is in the early stages of development. It also "depends" on a prototype data structure
package `datatree`. Both need to be installed from source.

```shell
pip install git+git://github.com/TomNicholas/datatree
pip install git+git://github.com/carbonplan/ndpyramid
```

# usage

Ndpyramid provides a set of utilites for creating pyramids with standardized metadata.
The example below demonstrates the usage of the `pyramid_coarsen` and `pyramid_reproject`
utilities.

```python
import xarray as xr
import rioxarray
from ndpyramid import pyramid_coarsen, pyramid_reproject

# load a sampel xarray.Dataset
ds = xr.tutorial.load_dataset('air_temperature')

# make a coarsened pyramid
pyramid = pyramid_coarsen(ds, factors=[16, 8, 4, 3, 2, 1], dims=['lat', 'lon'], boundary='trim')

# make a reprojected (EPSG:3857) pyramid
ds = ds.rio.write_crs('EPSG:4326')
pyramid = pyramid_reproject(ds, levels=2)

# write the pyramid to zarr
pyramid.to_zarr('./path/to/write')
```

See the docstrings for more details about input parameters and options.

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/) licensed, but we request that you please provide attribution if reusing any of our digital content (graphics, logo, articles, etc.).

## about us

CarbonPlan is a non-profit organization working on the science and data of carbon removal. We aim to improve the transparency and scientific integrity of carbon removal and climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/ndpyramid/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
