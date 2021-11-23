from collections import defaultdict
from typing import Dict, TypeVar

import datatree as dt
import xarray as xr

from .utils import get_version, multiscales_template

ResamplingType = TypeVar('ResamplingType', str, Dict[str, str])


def pyramid_reproject(
    ds,
    levels: int = None,
    pixels_per_tile=128,
    resampling: ResamplingType = 'average',
    extra_dim: str = None,
) -> dt.DataTree:
    """[summary]

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    levels : int, optional
        Number of levels in pyramid, by default None
    pixels_per_tile : int, optional
        Number of pixels to include along each axis in individual tiles, by default 128
    resampling : str or dict, optional
        Rasterio resampling method. Can be provided as a string or a per-variable
        dict, by default 'average'
    extra_dim : str, optional
        Extra dim by which to interate over, by default None

    Returns
    -------
    dt.DataTree
        Multiscale data pyramid
    """
    import rioxarray  # noqa: F401
    from rasterio.transform import Affine
    from rasterio.warp import Resampling

    # multiscales spec
    save_kwargs = {'levels': levels, 'pixels_per_tile': pixels_per_tile}
    attrs = {
        'multiscales': multiscales_template(
            datasets=[{'path': str(i)} for i in range(levels)],
            type='reduce',
            method='pyramid_reproject',
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    resampling_dict: ResamplingType
    if isinstance(resampling, str):
        resampling_dict = defaultdict(lambda: resampling)
    else:
        resampling_dict = resampling

    # set up pyramid
    root = xr.Dataset(attrs=attrs)
    pyramid = dt.DataTree(data_objects={"root": root})

    # pyramid data
    for level in range(levels):
        lkey = str(level)
        dim = 2 ** level * pixels_per_tile
        dst_transform = Affine.translation(-20026376.39, 20048966.10) * Affine.scale(
            (20026376.39 * 2) / dim, -(20048966.10 * 2) / dim
        )

        def reproject(da, var):
            return da.rio.reproject(
                'EPSG:3857',
                resampling=Resampling[resampling_dict[var]],
                shape=(dim, dim),
                transform=dst_transform,
            )

        pyramid[lkey] = xr.Dataset(attrs=ds.attrs)
        for k, da in ds.items():
            if len(da.shape) == 4:
                if extra_dim is None:
                    raise ValueError("must specify 'extra_dim' to iterate over 4d data")
                da_all = []
                for index in ds[extra_dim]:
                    da_reprojected = reproject(da.sel({extra_dim: index}), k)
                    da_all.append(da_reprojected)
                pyramid[lkey].ds[k] = xr.concat(da_all, ds[extra_dim])
            else:
                pyramid[lkey].ds[k] = reproject(da, k)
    return pyramid
