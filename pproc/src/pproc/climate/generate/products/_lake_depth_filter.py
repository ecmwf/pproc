# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Depth-mode-filter numerics for the ``lake-depth`` product.

DESCRIPTION: Aggregates a global 30''-resolution (~1 km) lake mean-depth
dataset onto a coarser regular lat/lon grid, using a source-dependent
statistic (MODE for GLDBv3 / geological approach, MEAN for GEBCO
bathymetry, DEFAULT 10 m elsewhere).

METHOD: aggregation of depth from different sources is made differently.

* IF DEPTH IS FROM GLDBv3 - aggregation method is MODE
* IF DEPTH IS ESTIMATED BY GEOLOGICAL APPROACH - aggregation method is MODE
* IF IT IS OCEAN BATHYMETRY (GEBCO) - aggregation method is MEAN
* IF NO IN-SITU OR ESTIMATED DEPTHS ARE AVAILABLE - no aggregation method,
  uses the DEFAULT DEPTH of 10m

OPTIMIZATIONS:

* Vectorized operations using NumPy for massive speedup
* Block-based processing to reduce Python loops
* Efficient bincount for MODE calculation
* Memory-efficient array views instead of copies

USED FILES: World_DEPTH.dat       - with lake mean depths in meters [REAL(4)]
            World_DEPTHStatus.dat - with sources of the depth data [INTEGER(1)]

AUTHORS : Margarita Choulga & Souhail Boussetta, 2017-09-20, ECMWF
Modified: Souhail Boussetta & Margarita Choulga, 2018-07-17, ECMWF (To output grib file)
Python conversion: A. van Niekerk 2026

Adapted for in-memory use in pproc.climate.generate.products.lake_depth
(the file-writing GRIB code and argparse entry point have been removed;
callers get back a numpy array and are responsible for GRIB encoding).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

__all__ = [
    "NLON_B",
    "NLAT_B",
    "STATUS_GLDBv3",
    "STATUS_GEBCO",
    "STATUS_GEO_APP",
    "STATUS_DEFAULT",
    "STATUS_GEBCO_GLDBv3",
    "STATUS_GEBCO_GEO_APP",
    "STATUS_GEBCO_DEFAULT",
    "GLDBv3_STATUSES",
    "DEPTH_GRADATION",
    "DEPTH_DEFAULT_VALUE",
    "read_depth_file",
    "read_status_file",
    "get_mode_depth_vectorized",
    "aggregate_depth_optimized",
    "compute_regridded_depth",
]


logger = logging.getLogger(__name__)


# Input file common parameters
NLON_B = (
    43200  # Number of longitude pixels of the bitmap (depth file) with resolution 1km
)
NLAT_B = 21600  # Number of latitude pixels of the bitmap with resolution 1km
PIX_SIZE = 30  # Pixel size of the bitmap in seconds of arc for resolution 1km
DEG_SIZE = 120  # Number of pixels of the bitmap in 1 degree for resolution 1km

# Explanation for different depth data sources
STATUS_GLDBv3 = (
    30  # lake mean depth information taken from GLDBv3 + newly added AralSea depths
)
STATUS_GEBCO = 73  # ocean bathymetry information taken from GEBCO
STATUS_GEO_APP = 75  # lake mean depth is estimated using geological approach from GLDB
STATUS_DEFAULT = 71  # default lake depth is used
STATUS_GEBCO_GLDBv3 = 33  # lake mean depth is calculated using GEBCO + GLDBv3
STATUS_GEBCO_GEO_APP = (
    35  # lake mean depth is calculated using GEBCO + geological approach depths
)
STATUS_GEBCO_DEFAULT = (
    31  # lake mean depth is calculated using GEBCO + default lake depth
)

# GLDBv3 status values
GLDBv3_STATUSES = np.array([1, 2, 3, 4, 5, 6, 7, 102, 103, 123, 124, 72], dtype=np.int8)

# Boundaries for depth gradations, m
DEPTH_GRADATION = np.array(
    [
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
        20.0,
        24.0,
        30.0,
        36.0,
        42.0,
        58.0,
        82.0,
        118.0,
        182.0,
        318.0,
        482.0,
        718.0,
        1282.0,
        99999.0,
    ]
)

# Mean depth for each depth gradation, m
DEPTH_DEFAULT_VALUE = np.array(
    [
        1.0,
        3.0,
        5.0,
        7.0,
        10.0,
        14.0,
        18.0,
        22.0,
        27.0,
        33.0,
        39.0,
        50.0,
        70.0,
        100.0,
        150.0,
        250.0,
        400.0,
        600.0,
        1000.0,
        1600.0,
    ]
)

ZMISS = 9999.0


def read_depth_file(
    filename: Path, nlon: int = NLON_B, nlat: int = NLAT_B
) -> np.ndarray:
    """Read depth data from binary file in REAL(4) format with little-endian encoding.

    Returns a Fortran-order (nlon, nlat) view over the raw bytes.
    """
    logger.info("Reading DEPTH file %s", filename)
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype="<f4", count=nlon * nlat)
    # Fortran array is World_Depth(NlonB, NlatB) - dimensions are (longitude, latitude)
    return data.reshape((nlon, nlat), order="F")


def read_status_file(
    filename: Path, nlon: int = NLON_B, nlat: int = NLAT_B
) -> np.ndarray:
    """Read status data from binary file in INTEGER(1) format with little-endian encoding.

    Returns a Fortran-order (nlon, nlat) view over the raw bytes.
    """
    logger.info("Reading STATUS file %s", filename)
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype="<i1", count=nlon * nlat)
    # Fortran array is World_Status(NlonB, NlatB) - dimensions are (longitude, latitude)
    return data.reshape((nlon, nlat), order="F")


def get_mode_depth_vectorized(
    depths: np.ndarray,
    statuses,  # unused; kept for signature compatibility with the reference
    depth_gradation: np.ndarray,
    depth_default_value: np.ndarray,
) -> Tuple[float, int]:
    """Vectorized calculation of MODE depth from depth values.

    Uses :func:`numpy.bincount` for efficient mode calculation.
    """
    # Find depth gradation indices using searchsorted
    indices = np.searchsorted(depth_gradation[1:], depths, side="left")

    # Count occurrences of each gradation
    counts = np.bincount(indices, minlength=20)

    # Find mode (most common gradation)
    if counts.sum() > 0:
        mode_idx = np.argmax(counts)
        return depth_default_value[mode_idx], counts.sum()
    else:
        return 0.0, 0


def aggregate_depth_optimized(
    world_depth: np.ndarray,
    world_status: np.ndarray,
    new_res: int,
    nlat_new: int,
    nlon_new: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized aggregation using vectorized NumPy operations.

    Parameters
    ----------
    world_depth:
        Original depth data, shape ``(nlon, nlat)`` in Fortran order.
    world_status:
        Original status data, shape ``(nlon, nlat)`` in Fortran order.
    new_res:
        Number of 1km pixels per new-resolution pixel (block-aggregation
        factor).
    nlat_new:
        Number of latitude pixels in the new resolution.
    nlon_new:
        Number of longitude pixels in the new resolution.

    Returns
    -------
    tuple of (ndarray, ndarray)
        ``(world_depth9, world_status9)`` — the aggregated depth and
        status arrays, both shaped ``(nlon_new, nlat_new)``.
    """
    logger.info("Starting optimized aggregation to a new resolution ...")

    depth_num_max = new_res * new_res
    # Fortran: World_Depth9(NlonB9, NlatB9) - dimensions are (longitude, latitude)
    world_depth9 = np.zeros((nlon_new, nlat_new), dtype=np.float32)
    world_status9 = np.zeros((nlon_new, nlat_new), dtype=np.int8)

    # Trim input arrays to fit evenly into new resolution
    nlat_trim = nlat_new * new_res
    nlon_trim = nlon_new * new_res

    # world_depth and world_status are (nlon, nlat), so index as [j, i] or [lon, lat]
    world_depth_trim = world_depth[:nlon_trim, :nlat_trim]
    world_status_trim = world_status[:nlon_trim, :nlat_trim]

    # Use a view-based approach to extract blocks without full reshape.
    # This matches the original loop structure exactly:
    #     for i in range(0, NLAT_B, new_res):  # latitude
    #         for j in range(0, NLON_B, new_res):  # longitude
    #             block = original[i:i+new_res, j:j+new_res]

    logger.info("Processing blocks ...")

    # Create masks for different data sources (vectorized for all data at once)
    mask_gldbv3_all = np.isin(world_status_trim, GLDBv3_STATUSES)
    mask_geo_app_all = world_status_trim == STATUS_GEO_APP
    mask_gebco_all = world_status_trim == STATUS_GEBCO
    mask_default_all = (world_status_trim == 0) | (world_status_trim == STATUS_DEFAULT)

    # Process each output pixel using slicing to extract blocks.
    # Fortran loops: do i=1,NlatB,NewRes (latitude), do j=1,NlonB,NewRes (longitude)
    # But Fortran arrays are World_Depth(j,i) - indexed as (longitude, latitude)
    for i9 in range(nlat_new):
        if i9 % 100 == 0:
            logger.info("Processing row %d/%d", i9, nlat_new)

        i_start = i9 * new_res
        i_end = i_start + new_res

        for j9 in range(nlon_new):
            j_start = j9 * new_res
            j_end = j_start + new_res

            # Extract block: arrays are (nlon, nlat) so index as [j, i] or [lon, lat].
            # This matches Fortran World_Depth((j+pj-1), (i+pi-1))
            block_depth = world_depth_trim[j_start:j_end, i_start:i_end]
            _ = world_status_trim[j_start:j_end, i_start:i_end]

            # Extract masks for this block
            gldbv3_mask = mask_gldbv3_all[j_start:j_end, i_start:i_end]
            geo_app_mask = mask_geo_app_all[j_start:j_end, i_start:i_end]
            gebco_mask = mask_gebco_all[j_start:j_end, i_start:i_end]
            default_mask = mask_default_all[j_start:j_end, i_start:i_end]

            # Count pixels
            depth_num_gldbv3 = int(np.sum(gldbv3_mask))
            depth_num_geo_app = int(np.sum(geo_app_mask))
            depth_num_gebco = int(np.sum(gebco_mask))
            depth_num_default = int(np.sum(default_mask))

            # Calculate depths from different sources
            depth_gldbv3 = 0.0
            if depth_num_gldbv3 > 0:
                depths_gldbv3 = block_depth[gldbv3_mask]
                depth_gldbv3, _ = get_mode_depth_vectorized(
                    depths_gldbv3, None, DEPTH_GRADATION, DEPTH_DEFAULT_VALUE
                )

            depth_geo_app = 0.0
            if depth_num_geo_app > 0:
                depths_geo_app = block_depth[geo_app_mask]
                depth_geo_app, _ = get_mode_depth_vectorized(
                    depths_geo_app, None, DEPTH_GRADATION, DEPTH_DEFAULT_VALUE
                )

            depth_gebco = 0.0
            if depth_num_gebco > 0:
                depth_gebco = float(np.mean(block_depth[gebco_mask]))

            depth_default = 10.0 if depth_num_default > 0 else 0.0

            # Check total
            total_pixels = (
                depth_num_gldbv3
                + depth_num_geo_app
                + depth_num_gebco
                + depth_num_default
            )
            if total_pixels != depth_num_max:
                logger.warning(
                    "Pixel mismatch at (%d,%d): expected %d, got %d",
                    i9,
                    j9,
                    depth_num_max,
                    total_pixels,
                )

            # Apply hierarchy scheme
            # Fortran: World_Depth9(j9,i9) - indexed as (longitude, latitude)
            if depth_gebco != 0.0:
                if depth_gldbv3 != 0.0:
                    world_depth9[j9, i9] = (
                        depth_gebco * depth_num_gebco + depth_gldbv3 * depth_num_gldbv3
                    ) / (depth_num_gebco + depth_num_gldbv3)
                    world_status9[j9, i9] = STATUS_GEBCO_GLDBv3
                elif depth_geo_app != 0.0:
                    world_depth9[j9, i9] = (
                        depth_gebco * depth_num_gebco
                        + depth_geo_app * depth_num_geo_app
                    ) / (depth_num_gebco + depth_num_geo_app)
                    world_status9[j9, i9] = STATUS_GEBCO_GEO_APP
                elif depth_num_default != 0:
                    world_depth9[j9, i9] = (
                        depth_gebco * depth_num_gebco
                        + depth_default * depth_num_default
                    ) / (depth_num_gebco + depth_num_default)
                    world_status9[j9, i9] = STATUS_GEBCO_DEFAULT
                else:
                    world_depth9[j9, i9] = depth_gebco
                    world_status9[j9, i9] = STATUS_GEBCO
            else:
                if depth_gldbv3 != 0.0:
                    world_depth9[j9, i9] = depth_gldbv3
                    world_status9[j9, i9] = STATUS_GLDBv3
                elif depth_geo_app != 0.0:
                    world_depth9[j9, i9] = depth_geo_app
                    world_status9[j9, i9] = STATUS_GEO_APP
                else:
                    world_depth9[j9, i9] = depth_default
                    world_status9[j9, i9] = STATUS_DEFAULT

    return world_depth9, world_status9


def compute_regridded_depth(
    world_depth_path: Path,
    world_depth_status_path: Path,
    new_res: int,
    nlat_new: int,
    nlon_new: int,
) -> np.ndarray:
    """Aggregate global 30'' depth data onto a coarser regular lat/lon grid.

    Reads the two source ``.dat`` files, runs
    :func:`aggregate_depth_optimized`, then applies the longitude shift
    that maps the source's (0..360°, prime meridian first) layout to the
    ECMWF-standard (–180..180°, mid-longitude first) layout expected by
    the GRIB template.

    Parameters
    ----------
    world_depth_path:
        Path to ``World_DEPTH.dat`` (43200 × 21600 little-endian
        ``float32``, Fortran order).
    world_depth_status_path:
        Path to ``World_DEPTHStatus.dat`` (43200 × 21600 little-endian
        ``int8``, Fortran order).
    new_res:
        Block-aggregation factor (1 km pixels per output pixel).
    nlat_new:
        Latitude points in the target regular lat/lon grid.
    nlon_new:
        Longitude points in the target regular lat/lon grid.

    Returns
    -------
    numpy.ndarray
        Flattened ``float32`` 1-D array of length ``nlon_new * nlat_new``
        ready for GRIB encoding against a matching regular-lat/lon
        template.
    """
    # Infer the source raster size from file length; the operational
    # inputs are 43200×21600, but the helper stays honest so tests can
    # feed synthetic smaller rasters.
    file_size = world_depth_path.stat().st_size
    # 4 bytes per float32. Derive nlon×nlat from the file size and the
    # operational aspect ratio (nlon = 2 × nlat).
    total = file_size // 4
    nlat_src = int(np.sqrt(total // 2))
    nlon_src = 2 * nlat_src
    if nlon_src * nlat_src * 4 != file_size:
        # Fall back to operational constants if inference does not match.
        nlon_src, nlat_src = NLON_B, NLAT_B

    logger.info(
        "compute_regridded_depth: source=%dx%d target=%dx%d block=%d",
        nlon_src,
        nlat_src,
        nlon_new,
        nlat_new,
        new_res,
    )

    world_depth = read_depth_file(world_depth_path, nlon_src, nlat_src)
    world_status = read_status_file(world_depth_status_path, nlon_src, nlat_src)

    world_depth9, _ = aggregate_depth_optimized(
        world_depth, world_status, new_res, nlat_new, nlon_new
    )

    # Prepare data for GRIB output: 1D longitude-shifted layout.
    # The source raster has longitude=0° at column 0 (0..360°). GRIB
    # regular-lat/lon templates used here start at –180°, so the second
    # half of each row needs to move to the front.
    logger.info("Transforming depth data into 1D array with longitude shift ...")
    south_pix = nlat_new
    east_pix = nlon_new
    north_pix = 0
    west_pix = 0
    lon_mid_pix = east_pix // 2

    read_vol = east_pix * south_pix
    separation_r = np.zeros(read_vol, dtype=np.float32)

    # Fortran loops: do i1=NorthPix,SouthPix (lat), do j1=WestPix,EastPix (lon).
    # But accesses World_Depth9(j1,i1) - (longitude, latitude).
    for i1 in range(north_pix, south_pix):
        for j1 in range(west_pix, east_pix):
            if j1 < lon_mid_pix:
                jj1 = j1 + lon_mid_pix
            else:
                jj1 = j1 - lon_mid_pix

            nk = i1 * east_pix + jj1
            # world_depth9 is (nlon, nlat), so index as [j1, i1].
            separation_r[nk] = world_depth9[j1, i1]

    logger.info("compute_regridded_depth done (%d points)", separation_r.size)
    return separation_r
