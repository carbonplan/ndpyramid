import importlib
from typing import List

import datatree as dt
import xarray as xr


def _get_version():
    try:
        return importlib.import_module('ndpyramid').__version__
    except ModuleNotFoundError:
        return '-9999'


def _multiscales_template(datasets=[], type='', method='', version='', args=[], kwargs={}):
    # https://forum.image.sc/t/multiscale-arrays-v0-1/37930
    d = [
        {
            "datasets": datasets,
            "type": type,
            "metadata": {"method": method, "version": version, "args": args, "kwargs": kwargs},
        }
    ]
    return d


def pyramid_coarsen(ds, factors: List[int], dims: List[str], **kwargs) -> dt.DataTree:

    # multiscales spec
    save_kwargs = locals()
    del save_kwargs['ds']

    attrs = {
        'multiscales': _multiscales_template(
            datasets=[{'path': str(i) for i in range(len(factors))}],
            type='reduce',
            method='pyramid_coarsen',
            version=_get_version(),
            kwargs=save_kwargs,
        )
    }

    # set up pyramid
    root = xr.Dataset(attrs=attrs)
    pyramid = dt.DataTree(data_objects={"root": root})

    # pyramid data
    for key, factor in enumerate(factors):

        skey = str(key)
        kwargs.update({d: factor for d in dims})
        pyramid[skey] = ds.coarsen(**kwargs).mean()

    return pyramid


def pyramid_reproject(
    ds, levels: int = None, pixels_per_tile=128, resampling='average', extra_dim=None
) -> dt.DataTree:
    import rioxarray  # noqa: F401
    from rasterio.transform import Affine
    from rasterio.warp import Resampling

    # multiscales spec
    save_kwargs = {'levels': levels, 'pixels_per_tile': pixels_per_tile}
    attrs = {
        'multiscales': _multiscales_template(
            datasets=[{'path': str(i) for i in range(levels)}],
            type='reduce',
            method='pyramid_reproject',
            version=_get_version(),
            kwargs=save_kwargs,
        )
    }

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

        def reproject(da):
            return da.rio.reproject(
                'EPSG:3857',
                resampling=Resampling[resampling],
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
                    da_reprojected = reproject(da.sel({extra_dim: index}))
                    da_all.append(da_reprojected)
                pyramid[lkey].ds[k] = xr.concat(da_all, ds[extra_dim])
            else:
                pyramid[lkey].ds[k] = reproject(da)
    return pyramid
