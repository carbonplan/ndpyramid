import numpy as np


def lon_to_180(ds):
    '''Converts longitude values to (-180, 180)

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with `lon` coordinate

    Returns
    -------
    xr.Dataset
        Copy of `ds` with updated coordinates

    See also
    --------
    cmip6_preprocessing.preprocessing.correct_lon

    Notes
    -----
    Copied from https://github.com/carbonplan/cmip6-downscaling
    '''

    ds = ds.copy()

    lon = ds['lon'].where(ds['lon'] < 180, ds['lon'] - 360)
    ds = ds.assign_coords(lon=lon)

    if not (ds['lon'].diff(dim='lon') > 0).all():
        ds = ds.reindex(lon=np.sort(ds['lon'].data))

    if 'lon_bounds' in ds.variables:
        lon_b = ds['lon_bounds'].where(ds['lon_bounds'] < 180, ds['lon_bounds'] - 360)
        ds = ds.assign_coords(lon_bounds=lon_b)

    return ds
