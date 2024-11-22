import typing

import pydantic
import pyproj
import rasterio.transform

ProjectionOptions = typing.Literal["web-mercator", "equidistant-cylindrical"]


class Projection(pydantic.BaseModel):
    name: ProjectionOptions = "web-mercator"
    _crs: str = pydantic.PrivateAttr()
    _proj = pydantic.PrivateAttr()

    def __init__(self, **data) -> None:

        super().__init__(**data)
        epsg_codes = {"web-mercator": "EPSG:3857", "equidistant-cylindrical": "EPSG:4326"}
        # Area extent for pyresample's `create_area_def` is (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
        area_extents = {
            "web-mercator": (
                -20037508.342789244,
                -20037508.342789248,
                20037508.342789244,
                20037508.342789248,
            ),
            "equidistant-cylindrical": (-180, 90, 180, -90),
        }
        self._crs = epsg_codes[self.name]
        self._proj = pyproj.Proj(self._crs)
        self._area_extent = area_extents[self.name]

    @pydantic.validate_call
    def transform(self, *, dim: int) -> rasterio.transform.Affine:

        if self.name == "web-mercator":
            # set up the transformation matrix for the web-mercator projection such that the data conform
            # to the slippy-map tiles assumed boundaries. See https://github.com/carbonplan/ndpyramid/pull/70
            # for detailed on calculating the parameters.
            return rasterio.transform.Affine.translation(
                -20037508.342789244, 20037508.342789248
            ) * rasterio.transform.Affine.scale(
                (20037508.342789244 * 2) / dim, -(20037508.342789248 * 2) / dim
            )
        elif self.name == "equidistant-cylindrical":
            # set up the transformation matrix that maps between the Equidistant Cylindrical projection
            # and the latitude-longitude projection. The Affine.translation function moves the origin
            # of the grid from (0, 0) to (-180, 90) in latitude-longitude coordinates,
            # and the Affine.scale function scales the grid coordinates to match the size of the grid
            # in latitude-longitude coordinates. The resulting transformation matrix maps grid coordinates to
            # latitude-longitude coordinates.
            return rasterio.transform.Affine.translation(
                -180, 90
            ) * rasterio.transform.Affine.scale(360 / dim, -180 / dim)
