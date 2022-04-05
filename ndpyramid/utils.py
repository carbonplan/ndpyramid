from __future__ import annotations  # noqa: F401

import contextlib

import datatree as dt
import numpy as np
import numpy.typing as npt
import xarray as xr

from ._version import __version__

# from netCDF4 and netCDF4-python
default_fillvals = {
    'S1': '\x00',
    'i1': -127,
    'u1': 255,
    'i2': -32767,
    'u2': 65535,
    'i4': -2147483647,
    'u4': 4294967295,
    'i8': -9223372036854775806,
    'u8': 18446744073709551614,
    'f4': 9.969209968386869e36,
    'f8': 9.969209968386869e36,
}


def get_version() -> str:
    return __version__


def multiscales_template(
    datasets: list = None,
    type: str = '',
    method: str = '',
    version: str = '',
    args: list = None,
    kwargs: dict = None,
):
    if datasets is None:
        datasets = []
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    # https://forum.image.sc/t/multiscale-arrays-v0-1/37930
    return [
        {
            'datasets': datasets,
            'type': type,
            'metadata': {'method': method, 'version': version, 'args': args, 'kwargs': kwargs},
        }
    ]


def set_zarr_encoding(
    ds: xr.Dataset,
    codec_config: dict | None = None,
    float_dtype: npt.DTypeLike | None = None,
    int_dtype: npt.DTypeLike | None = None,
) -> xr.Dataset:
    """Set zarr encoding for each variable in the dataset

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    codec_config : dict, optional
        Dictionary of parameters to pass to numcodecs.get_codec.
        The default is {'id': 'zlib', 'level': 1}
    float_dtype : str or dtype, optional
        Dtype to cast floating point variables to

    Returns
    -------
    ds : xr.Dataset
        Output dataset with updated variable encodings
    """
    import numcodecs

    ds = ds.copy()

    if codec_config is None:
        codec_config = {'id': 'zlib', 'level': 1}
    compressor = numcodecs.get_codec(codec_config)

    for k, da in ds.variables.items():

        # maybe cast float type
        if np.issubdtype(da.dtype, np.floating) and float_dtype is not None:
            da = da.astype(float_dtype)

        if np.issubdtype(da.dtype, np.integer) and int_dtype is not None:
            da = da.astype(int_dtype)

        # remove old encoding
        da.encoding.clear()

        # update with new encoding
        da.encoding['compressor'] = compressor
        with contextlib.suppress(AttributeError):
            del da.atrrs['_FillValue']
        da.encoding['_FillValue'] = default_fillvals.get(
            da.dtype.str[-2:], None
        )  # TODO: handle date/time types

        ds[k] = da

    return ds


def _add_metadata_and_zarr_encoding(
    dt: dt.DataTree, *, levels: int, other_chunks: dict = None, pixels_per_tile: int = 128
) -> dt.DataTree:

    '''Postprocess data pyramid. Adds multiscales metadata and sets Zarr encoding

    Parameters
    ----------
    dt : dt.DataTree
        Input data pyramid
    levels : int
        Number of levels in pyramid
    other_chunks : dict
        Chunks for non-spatial dims
    pixels_per_tile : int
        Number of pixels per tile

    Returns
    -------
    dt.DataTree
        Updated data pyramid with metadata / encoding set
    '''
    chunks = {'x': pixels_per_tile, 'y': pixels_per_tile}
    if other_chunks is not None:
        chunks.update(other_chunks)

    for level in range(levels):
        slevel = str(level)
        dt.ds.attrs['multiscales'][0]['datasets'][level]['pixels_per_tile'] = pixels_per_tile

        # set dataset chunks
        dt[slevel].ds = dt[slevel].ds.chunk(chunks)
        if 'date_str' in dt[slevel].ds:
            dt[slevel].ds['date_str'] = dt[slevel].ds['date_str'].chunk(-1)

        # set dataset encoding
        dt[slevel].ds = set_zarr_encoding(
            dt[slevel].ds, codec_config={'id': 'zlib', 'level': 1}, float_dtype='float32'
        )
        for var in ['time', 'time_bnds']:
            if var in dt[slevel].ds:
                dt[slevel].ds[var].encoding['dtype'] = 'int32'

    # set global metadata
    dt.ds.attrs.update({'title': 'multiscale data pyramid', 'ndpyramid_version': __version__})
    return dt
