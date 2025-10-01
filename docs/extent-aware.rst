Extent-Aware (Cropped) Pyramids
================================

.. note::
   This feature is experimental. The API and metadata fields may change in a
   minor release while we gather feedback.

Overview
--------

By default, ``ndpyramid`` generates *global* multiscale levels: each zoom level
is a square array whose dimension is ``2**level * pixels_per_tile`` covering
the entire projection extent (Web Mercator or Equidistant Cylindrical). For
regional datasets this can allocate very large, mostly empty arrays and create
large Dask graphs.

The *extent-aware* ("cropped") mode builds only the tiles that intersect the
true data footprint. Each level is still aligned to the implicit global tile
matrix (slippy-map style), but the stored arrays are smaller rectangles:

.. code-block:: text

   global level (conceptual)      cropped level actually stored
   ---------------------------    ------------------------------
   |  world pixels ...       |    |  only tiles intersecting   |
   |  (many empty tiles)     |    |  the regional extent       |
   ---------------------------    ------------------------------

Benefits:

* Dramatically reduced memory & compute for regional data (proportional to
  area fraction of world).
* Fewer Dask tasks – faster graph scheduling.
* Exact tile/grid alignment preserved for downstream tile addressing.

Usage
-----

Pass ``crop=True`` to ``pyramid_reproject`` (and optionally an explicit
``extent``):

.. code-block:: python

   from ndpyramid import pyramid_reproject
   from odc.geo.xr import assign_crs

   # ds has lon/lat or already projected coordinates
   ds = assign_crs(ds, "EPSG:4326")

   pyr_cropped = pyramid_reproject(
       ds,
       levels=6,
       projection="web-mercator",
       pixels_per_tile=128,
       crop=True,               # activate extent-aware mode
       # extent=(xmin, ymin, xmax, ymax),  # optional; in target CRS if provided
   )

   pyr_cropped.to_zarr("cropped-pyramid.zarr", mode="w", consolidated=True)

If ``extent`` is not supplied we infer it from the ``x`` and ``y`` coordinate
ranges of the dataset (after any reprojection upstream). The inferred extent is
clamped to the full projection bounds.

Metadata Additions
------------------

For cropped levels, each dataset entry inside ``multiscales[0]['datasets']``
contains a nested ``geospatial`` object with:

================  =============================================================
Field             Description
================  =============================================================
``tile_offset``   ``[tile_x0, tile_y0]`` global tile indices of top-left tile
``tile_shape``    ``[tiles_x, tiles_y]`` count of tiles stored in x and y
``shape_px``      ``[height_px, width_px]`` pixel dimensions of stored array
``resolution``    ``[res_x, res_y]`` pixel size (usually square) at this level
``extent_full``   Full projection extent (world) in target CRS
``extent_data``   Tight data bbox (before outward snapping to tile grid)
``transform``     Affine transform (raster -> CRS) ``[a,b,c,d,e,f]``
``layout``        Always ``"cropped"`` for cropped levels
================  =============================================================

Example ``multiscales`` snippet:

.. code-block:: json

   {
     "multiscales": [
       {
         "datasets": [
           {
             "path": "5",
             "level": 5,
             "crs": "EPSG:3857",
             "geospatial": {
               "tile_offset": [301, 384],
               "tile_shape": [22, 18],
               "shape_px": [2304, 2816],
               "resolution": [38.2185, 38.2185],
               "extent_full": [-20037508.34, -20037508.34, 20037508.34, 20037508.34],
               "extent_data": [-13760000.0, 2800000.0, -7400000.0, 6350000.0],
               "transform": [-13762500.0, 0.0, 38.2185, 6350312.0, -38.2185, 0.0],
               "layout": "cropped"
             }
           }
         ],
         "metadata": {"method": "pyramid_reproject", "kwargs": {"crop": true}},
         "type": "reduce"
       }
     ]
   }

(Values above illustrative only.)

How Tile Offsets Work
---------------------

The global tile matrix for level *Z* has dimensions ``2**Z`` x ``2**Z`` tiles.
``tile_offset`` tells you where the stored submatrix begins. A map client that
needs tile ``(Z, x, y)`` checks if

* ``x`` in ``[tile_x0, tile_x0 + tiles_x - 1]`` and
* ``y`` in ``[tile_y0, tile_y0 + tiles_y - 1]``.

If true, the data lives inside the cropped array at:

.. code-block:: text

   local_x = x - tile_x0
   local_y = y - tile_y0
   x pixel slice = local_x * pixels_per_tile : (local_x + 1) * pixels_per_tile
   y pixel slice = local_y * pixels_per_tile : (local_y + 1) * pixels_per_tile

Relationship to Full-World Pyramids
-----------------------------------

Pixel resolutions across zoom levels are identical to the full-world scheme;
we only omit tiles outside the region. Any analysis expecting global arrays can
still reconstruct conceptual coordinates using ``tile_offset``, ``tile_shape``
and ``resolution``.

Limitations / Future Work
-------------------------

* ``extent_crs`` parameter is reserved; currently the passed ``extent`` (if
  any) is interpreted in the target projection.
* Masking of partial tiles outside the original (non-snapped) bbox is not yet
  performed – edge pixels beyond the data bbox may contain nodata from the
  reprojection step.
* Only ``pyramid_reproject`` supports cropping in this release; ``pyramid_resample``
  may gain it later.

Best Practices
--------------

* Choose a ``pixels_per_tile`` matching downstream tile size (128–512 typical).
* Benchmark memory usage by comparing sum of ``shape_px`` across levels versus
  old full-world allocation.
* Keep a record of the original (uncropped) bounding box if you plan to union
  multiple cropped pyramids later.

Changelog Notes
---------------

If you publish cropped pyramids, note the presence of ``geospatial`` metadata
so consumers can adapt. Downstream clients that ignore it still function (they
just see smaller arrays), but cannot place them on a map without it.
