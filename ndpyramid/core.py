from typing import List

import datatree as dt
import xarray as xr

from .utils import get_version, multiscales_template


def pyramid_coarsen(ds, factors: List[int], dims: List[str], **kwargs) -> dt.DataTree:

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
    root = xr.Dataset(attrs=attrs)
    pyramid = dt.DataTree(data=root, name='root')

    # pyramid data
    for key, factor in enumerate(factors):

        skey = str(key)
        kwargs.update({d: factor for d in dims})
        pyramid[skey] = ds.coarsen(**kwargs).mean()

    return pyramid
