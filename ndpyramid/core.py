from typing import Any, Dict, List, MutableMapping

import numpy as np
import xarray as xr


class BasePyramid:

    obj: xr.Dataset
    pyramid_opts: Dict[str, Any]
    pyramid: Dict[str, xr.Dataset]
    schema: Dict[str, Any]

    def __init__(self, obj: xr.Dataset, **kwargs) -> None:
        self.obj = obj
        self.pyramid_opts = kwargs

        self.make_pyramid(**self.pyramid_opts)

    def make_pyramid(self, **kwargs) -> None:
        # self.schema = {}
        # self.pyramid = {}
        raise NotImplementedError()

    def to_zarr(self, store: MutableMapping = None) -> MutableMapping:
        import zarr

        with zarr.group(store) as root:
            root.attrs.update(self.schema)

        for level, ds in self.pyramid.items():
            ds.to_zarr(store, consolidated=False, group=level)

        return store

    def __repr__(self) -> str:
        t = type(self)
        r = f'<{t.__module__}.{t.__name__}>'
        return r

    def __str__(self) -> str:
        return repr(self)


class XarrayCoarsenPyramid(BasePyramid):
    def make_pyramid(self, levels: int = None, dims: List[str] = None, **kwargs) -> None:
        self.schema = {}
        self.pyramid = {}

        factors = zip(np.arange(levels)[::-1], 2 ** np.arange(levels))

        for key, factor in factors:
            level = str(key)
            kwargs.update({d: factor for d in dims})
            self.pyramid[level] = self.obj.coarsen(**kwargs).mean()
