from __future__ import annotations  # noqa: F401

from collections import defaultdict

import datatree as dt
import xarray as xr

from .utils import add_metadata_and_zarr_encoding, get_version, multiscales_template


def pyramid_coarsen(
    ds: xr.Dataset, *, factors: list[int], dims: list[str], **kwargs
) -> dt.DataTree:
    """Create a multiscale pyramid via coarsening of a dataset by given factors

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to coarsen.
    factors : list[int]
        The factors to coarsen by.
    dims : list[str]
        The dimensions to coarsen.
    kwargs : dict
        Additional keyword arguments to pass to xarray.Dataset.coarsen.
    """

    # multiscales spec
    save_kwargs = locals()
    del save_kwargs['ds']

    attrs = {
        'multiscales': multiscales_template(
            datasets=[{'path': str(i)} for i in range(len(factors))],
            type='reduce',
            method='pyramid_coarsen',
            version=get_version(),
            kwargs=save_kwargs,
        )
    }

    # set up pyramid
    plevels = {}

    # pyramid data
    for key, factor in enumerate(factors):
        # merge dictionary via union operator
        kwargs |= {d: factor for d in dims}
        plevels[str(key)] = ds.coarsen(**kwargs).mean()

    plevels['/'] = xr.Dataset(attrs=attrs)
    return dt.DataTree.from_dict(plevels)


def pyramid_reproject(
    ds: xr.Dataset,
    *,
    levels: int = None,
    pixels_per_tile: int = 128,
    other_chunks: dict = None,
    resampling: str | dict = 'average',
    extra_dim: str = None,
) -> dt.DataTree:

    """Create a multiscale pyramid of a dataset via reprojection.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to create a multiscale pyramid of.
    levels : int, optional
        The number of levels to create. If None, the number of levels is
        determined by the number of tiles in the dataset.
    pixels_per_tile : int, optional
        Number of pixels per tile, by default 128
    other_chunks : dict
        Chunks for non-spatial dims to pass to :py:meth:`~xr.Dataset.chunk`. Default is None
    resampling : str or dict, optional
        Rasterio warp resampling method to use. Default is 'average'.
        If a dict, keys are variable names and values are warp resampling methods.
    extra_dim : str, optional
        The name of the extra dimension to iterate over. Default is None.

    Returns
    -------
    dt.DataTree
        The multiscale pyramid.

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

    # Convert resampling from string to dictionary if necessary
    if isinstance(resampling, str):
        resampling_dict = defaultdict(lambda: resampling)
    else:
        resampling_dict = resampling

    # set up pyramid
    plevels = {}

    # pyramid data
    for level in range(levels):
        lkey = str(level)
        dim = 2**level * pixels_per_tile
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

        # create the data array for each level
        plevels[lkey] = xr.Dataset(attrs=ds.attrs)
        for k, da in ds.items():
            if len(da.shape) == 4:
                # if extra_dim is not specified, raise an error
                if extra_dim is None:
                    raise ValueError("must specify 'extra_dim' to iterate over 4d data")
                da_all = []
                for index in ds[extra_dim]:
                    # reproject each index of the 4th dimension
                    da_reprojected = reproject(da.sel({extra_dim: index}), k)
                    da_all.append(da_reprojected)
                plevels[lkey][k] = xr.concat(da_all, ds[extra_dim])
            else:
                # if the data array is not 4D, just reproject it
                plevels[lkey][k] = reproject(da, k)

    # create the final multiscale pyramid
    plevels['/'] = xr.Dataset(attrs=attrs)
    pyramid = dt.DataTree.from_dict(plevels)

    pyramid = add_metadata_and_zarr_encoding(
        pyramid, levels=levels, pixels_per_tile=pixels_per_tile, other_chunks=other_chunks
    )
    return pyramid
