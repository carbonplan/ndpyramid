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

    def astype(self, dtype, **kwargs):
        for key, ds in self.pyramid.items():
            self.pyramid[key] = ds.astype(dtype, **kwargs)
        return self   

    def chunk(self, **kwargs):
        for key, ds in self.pyramid.items():
            self.pyramid[key] = ds.chunk(**kwargs)
        return self

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


class ReprojectedPyramid(BasePyramid):
    def make_pyramid(self, levels: int = None, pixels_per_tile=128) -> None:
        from rasterio.transform import Affine
        from rasterio.warp import Resampling
        
        self.schema = {}
        self.pyramid = {}

        for level in range(levels):
            dim = 2 ** level * pixels_per_tile
            dst_transform = Affine.translation(-20026376.39, 20048966.10) * Affine.scale((20026376.39*2)/dim, -(20048966.10*2)/dim)

            self.pyramid[str(level)] = xr.Dataset(attrs=self.obj.attrs)
            for k, da in self.obj.items():
                self.pyramid[str(level)][k] = da.rio.reproject(
                    'EPSG:3857', resampling=Resampling.average, shape=(dim, dim), transform=dst_transform)
