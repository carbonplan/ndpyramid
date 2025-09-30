from __future__ import annotations

"""Helpers for extent-aware (cropped) pyramid layouts.

The goal is to compute, for each requested zoom level, a destination GeoBox
covering only the tiles that intersect the data footprint while remaining
aligned to the implicit global slippy-map tile matrix (origin top-left, Web
Mercator by default).

This module intentionally keeps logic independent from the reproject/resample
implementations so future writers (e.g., tile-object storage) can reuse the
same math.
"""

import math
from dataclasses import dataclass

from affine import Affine
from odc.geo import CRS as OdcCRS
from odc.geo.geobox import GeoBox


@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def normalize(self) -> BBox:  # noqa: D401
        """Return a normalized BBox with ordered coordinates."""
        xmin, xmax = sorted((self.xmin, self.xmax))
        ymin, ymax = sorted((self.ymin, self.ymax))
        return BBox(xmin, ymin, xmax, ymax)


class PyramidLayout:
    """Computes cropped GeoBoxes for pyramid levels.

    Parameters
    ----------
    world_extent : tuple[float,float,float,float]
        (xmin, ymin, xmax, ymax) of the full projection extent.
    pixels_per_tile : int
        Tile edge in pixels.
    crs : str | pyproj.CRS (string form used by odc-geo)
        Target CRS.
    """

    def __init__(
        self, *, world_extent: tuple[float, float, float, float], pixels_per_tile: int, crs: str
    ):
        xmin, ymin, xmax, ymax = world_extent
        # ensure ordering
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        self.world_extent = (xmin, ymin, xmax, ymax)
        self.pixels_per_tile = pixels_per_tile
        self.crs = crs
        self._world_width = xmax - xmin
        self._world_height = ymax - ymin

    # ------------------------------------------------------------------
    # Resolution ladder
    # ------------------------------------------------------------------
    def pixel_size(self, level: int) -> float:
        """Pixel size (square) at a given level for the *full* world grid.

        The canonical rule mirrors existing implementation where a full
        world array would have dimension = 2**level * pixels_per_tile.
        """

        dim_full = (2**level) * self.pixels_per_tile
        return self._world_width / dim_full

    # ------------------------------------------------------------------
    def level_from_bbox(self, *, level: int, bbox: BBox) -> dict[str, object]:
        """Return cropped GeoBox + metadata for a level.

        Returns
        -------
        dict with keys:
          geobox : odc.geo.GeoBox
          metadata : dict (tile_offset, tile_shape, resolution, transform,
                           extent_data, extent_full, shape_px)
        """
        bbox_n = bbox.normalize()
        wxmin, wymin, wxmax, wymax = self.world_extent
        # clamp bbox to world
        xmin = max(wxmin, bbox_n.xmin)
        xmax = min(wxmax, bbox_n.xmax)
        ymin = max(wymin, bbox_n.ymin)
        ymax = min(wymax, bbox_n.ymax)

        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Invalid / empty intersection between data bbox and world extent")

        pixel = self.pixel_size(level)
        tile_world = self.pixels_per_tile * pixel

        # tile indices (origin: top-left, y inverted)
        # y origin at world ymax for slippy-style indexing
        wymax_origin = wymax
        tile_x_min = math.floor((xmin - wxmin) / tile_world)
        tile_x_max = math.floor((xmax - wxmin - 1e-9) / tile_world)
        tile_y_min = math.floor((wymax_origin - ymax) / tile_world)
        tile_y_max = math.floor((wymax_origin - ymin - 1e-9) / tile_world)

        tiles_x = tile_x_max - tile_x_min + 1
        tiles_y = tile_y_max - tile_y_min + 1

        width_px = tiles_x * self.pixels_per_tile
        height_px = tiles_y * self.pixels_per_tile

        xmin_snap = wxmin + tile_x_min * tile_world
        ymax_snap = wymax_origin - tile_y_min * tile_world

        # Affine: translation to upper-left corner then scale
        transform = Affine.translation(xmin_snap, ymax_snap) * Affine.scale(pixel, -pixel)

        geobox = GeoBox((height_px, width_px), transform, OdcCRS(self.crs))

        ta, tb, tc, td, te, tf = tuple(transform)[:6]
        metadata: dict[str, object] = {
            "tile_offset": [int(tile_x_min), int(tile_y_min)],
            "tile_shape": [int(tiles_x), int(tiles_y)],
            "shape_px": [int(height_px), int(width_px)],
            "resolution": [float(pixel), float(pixel)],
            "extent_full": [wxmin, wymin, wxmax, wymax],
            "extent_data": [xmin, ymin, xmax, ymax],
            "transform": [ta, tb, tc, td, te, tf],
            "layout": "cropped",
        }

        return {"geobox": geobox, "metadata": metadata}


__all__ = ["PyramidLayout", "BBox"]
