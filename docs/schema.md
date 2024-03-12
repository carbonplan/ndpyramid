# Schema

## Background

`ndpyramid` was created to generate pyramids for use with the [`@carbonplan/maps`](https://github.com/carbonplan/maps) toolkit. Check out blog posts about the [initial release](https://carbonplan.org/blog/maps-library-release) and [updates](https://carbonplan.org/blog/zarr-visualization-update) for more information about the toolkit's history. While our mapping toolkit remains the primary motivation for the library, the pyramids also have the potential to speed up other mapping approaches such as [dynamic tiling](https://nasa-impact.github.io/zarr-visualization-report/approaches/tiling/05-cmip6-pyramids.html).

In order to provide highly performant rendering, the `pyramid_reproject` and `pyramid_regrid` methods generate pyramids according to the [map zoom level quadtree](https://docs.mapbox.com/help/glossary/zoom-level/#zoom-level-quadtrees) pattern. This structure, in which the number of tiles at a given zoom level corresponds to 2<sup>zoom</sup> and each zoom level covers the entire globe, is commonly referred to as ['web-optimized'](https://cogeotiff.github.io/rio-cogeo/Advanced/#web-optimized-cog) when levels are in the Web Mercator projection because it minimizes the number of GET requests required for rendering a tile map.

In fact, earlier releases of the toolkit only generated and supported Web Mercator (EPSG:3857) pyramids, both to minimize GET requests and avoid reprojection on the client. Release `v3.0.0` of `@carbonplan/maps` and `v0.1.0` of `ndpyramid` added support for the Equidistant Cylindrical (EPSG:4326) projection for cases in which users want pyramids in the same projection as the original data, at the [expense of slower rendering times](https://nasa-impact.github.io/zarr-visualization-report/approaches/dynamic-client/e2e-results-projection.html).

## Pyramid schema

While the [map zoom level quadtree structure](https://docs.mapbox.com/help/glossary/zoom-level/#zoom-level-quadtrees) has been used for many years, there was no convention for storing the quadtree pyramids in Xarray and Zarr when we started work on this toolkit (although parallel development occurred in the microscopy and other communities). Therefore, we created a pyramid and metadata schema for `ndpyramid`. The resulting Zarr store for a dataset with one `tavg` data variable would look like:

```{code}
/
 ├── .zmetadata
 ├── 0
 │   ├── tavg
 │       └── 0.0
 ├── 1
 │   ├── tavg
 │       └── 0.0
 │       └── 0.1
 │       └── 1.0
 │       └── 1.1
 ├── 2
...
```

Note the quadrupling of the number of chunks as zoom level increases. This, combined with the global extent of individual levels and specific projection, allows inference of the placement of chunks on a web map based on the chunk index.

Metadata about the pyramids is stored in the `multiscales` attribute of the Xarray DataTree or Zarr store:

```{code}
{
  "multiscales": [
    {
      "datasets": [
        {
          "path": "0",
          "pixels_per_tile": 128,
          "crs": "EPSG:3857"
        },
        {
          "path": "1",
          "pixels_per_tile": 128,
          "crs": "EPSG:3857"
        }
        ...
      ],
      "metadata": {
        "args": [],
        "method": "pyramid_reproject",
        "version": "0.0.post64"
      },
      "type": "reduce"
    }
  ]
}
```

Currently, `@carbonplan/maps` does not rely on the `"crs"` attribute, but future releases may determine the projection based on that attribute (assuming Web Mercator projection if it is not provided).

In addition, the mapping toolkit relies on the `_ARRAY_DIMENSIONS` attribute introduced by Xarray, which stores the dimension names.

## Pyramids for @carbonplan/maps

In addition to following the quadtree pyramid structure and metadata schema, the pyramids currently must also meet the following requirements for use with `@carbonplan/maps`:

- Consistent chunk size across pyramid levels (128, 256, or 512 are recommended)
- Storage of non-spatial coordinate arrays in single chunk
- [zlib](https://numcodecs.readthedocs.io/en/stable/zlib.html) or [gzip](https://numcodecs.readthedocs.io/en/stable/gzip.html) compression
- Web Mercator (EPSG:3857) or Equidistant Cylindrical (EPSG:4326) projection
- The `.zattrs` must conform to the [IETF JSON Standard](https://datatracker.ietf.org/doc/html/rfc8259).
- Data types supported by [zarr-js](https://github.com/freeman-lab/zarr-js). The following are supported as of `v3.3.0` for Zarr v2:

  ```{code}
  '<i1': Int8Array,
  '<u1': Uint8Array,
  '|b1': BoolArray,
  '|u1': Uint8Array,
  '<i2': Int16Array,
  '<u2': Uint16Array,
  '<i4': Int32Array,
  '<u4': Uint32Array,
  '<f4': Float32Array,
  '<f8': Float64Array,
  '<U': StringArray,
  '|S': StringArray,
  ```

We recommend exploring the [`@carbonplan/maps` repository](https://github.com/carbonplan/maps), [`@carbonplan/maps` documentation](https://docs.carbonplan.org/maps), and [Zarr visualization report](https://nasa-impact.github.io/zarr-visualization-report/) for more information about CarbonPlan's approach to interactive multi-dimensional data-driven web maps.
