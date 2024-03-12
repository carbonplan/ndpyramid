<p align="left" >
<a href='https://carbonplan.org'>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://carbonplan-assets.s3.amazonaws.com/monogram/light-small.png">
  <img alt="CarbonPlan monogram." height="48" src="https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png">
</picture>
</a>
</p>

# ndpyramid

A small utility for generating ND array pyramids using Xarray and Zarr.

[![CI](https://github.com/carbonplan/ndpyramid/actions/workflows/main.yaml/badge.svg)](https://github.com/carbonplan/ndpyramid/actions/workflows/main.yaml)
![PyPI](https://img.shields.io/pypi/v/ndpyramid)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ndpyramid.svg)](https://anaconda.org/conda-forge/ndpyramid)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# installation

Ndpyramid can be installed in three ways:

Using the [conda](https://conda.io) package manager that comes with the Anaconda/Miniconda distribution:

```shell
conda install ndpyramid --channel conda-forge
```

Using the [pip](https://pypi.org/project/pip/) package manager:

```shell
python -m pip install ndpyramid
```

To install a development version from source:

```python
git clone https://github.com/carbonplan/ndpyramid
cd ndpyramid
python -m pip install -e .
```

# usage

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

See the docstrings for more details about input parameters and options.

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/)-licensed, but we request that you please provide attribution if reusing any of our digital content (graphics, logo, articles, etc.).

## about us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions with open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/ndpyramid/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
