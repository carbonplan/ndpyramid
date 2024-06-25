# Installation

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

## Optional dependencies

Depending on your use case you can specify optional dependencies on install.

```
python -m pip install "ndpyramid[xesmf]"     # Install optional dependencies for regridding with ESMF
python -m pip install "ndpyramid[dask]"      # Install optional dependencies for resampling with pyresample and Dask
python -m pip install "ndpyramid[complete]"  # Install all optional dependencies
```
