import typing

import pydantic
import pyproj
import rasterio.transform


class Projection(pydantic.BaseModel):
    name: typing.Literal['web-mercator', 'equidistant-cylindrical'] = 'web-mercator'
    _crs: str = pydantic.PrivateAttr()
    _proj = pydantic.PrivateAttr()

    def __init__(self, **data) -> None:

        super().__init__(**data)
        epsg_codes = {'web-mercator': 'EPSG:3857', 'equidistant-cylindrical': 'EPSG:4326'}
        self._crs = epsg_codes[self.name]
        self._proj = pyproj.Proj(self._crs)

    @pydantic.validate_arguments
    def transform(self, *, dim:int) -> rasterio.transform.Affine:


        if self.name == 'web-mercator':
            return rasterio.transform.Affine.translation(-20026376.39, 20048966.10) * rasterio.transform.Affine.scale((20026376.39 * 2) / dim, -(20048966.10 * 2) / dim)
        elif self.name == 'equidistant-cylindrical':
            # set up the transformation matrix that maps between the Equidistant Cylindrical projection
            # and the latitude-longitude projection. The Affine.translation function moves the origin
            # of the grid from (0, 0) to (-180, 90) in latitude-longitude coordinates,
            # and the Affine.scale function scales the grid coordinates to match the size of the grid
            # in latitude-longitude coordinates. The resulting transformation matrix maps grid coordinates to
            # latitude-longitude coordinates.
            return rasterio.transform.Affine.translation(-180, 90) * rasterio.transform.Affine.scale(360 / dim, -180 / dim)
