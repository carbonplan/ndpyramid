import mercantile
import numpy as np
import xarray as xr


def _bounds(ds):
    left = ds.x[0] - (ds.x[1] - ds.x[0]) / 2
    right = ds.x[-1] + (ds.x[-1] - ds.x[-2]) / 2
    top = ds.y[0] + (ds.y[1] - ds.y[0]) / 2
    bottom = ds.y[-1] - (ds.y[-1] - ds.y[-2]) / 2
    return (left.data, bottom.data, right.data, top.data)


def verify_xy_bounds(ds, zoom):
    """
    Verifies that the bounds of a chunk conforms to expectations for a WebMercatorQuad.
    """
    tile = mercantile.tile(ds.x[0], ds.y[0], zoom)
    expected = mercantile.xy_bounds(tile)
    np.testing.assert_allclose(expected, _bounds(ds))
    return ds


def verify_bounds(pyramid):
    for level in pyramid:
        if pyramid[level].attrs['multiscales'][0]['datasets'][0]['crs'] == 'EPSG:3857':
            xr.map_blocks(
                verify_xy_bounds,
                pyramid[level].ds,
                template=pyramid[level].ds,
                kwargs={'zoom': int(level)},
            )
        else:
            raise ValueError('Tile boundary verification has only been implemented for EPSG:3857')
