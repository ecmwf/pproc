# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``lake-depth`` product: aggregate 30''-resolution depth data to a target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_lake_depth.ksh``. Steps
3-6 of the legacy pipeline are folded into this product:

1. **Step 3** — mode/mean/default aggregation of ``World_DEPTH.dat`` +
   ``World_DEPTHStatus.dat`` onto a regular lat/lon grid whose resolution
   is passed in as ``resol_km`` x ``nlat`` x ``nlon`` (previously done by
   the Fortran ``depth_mode_filter`` binary; now :mod:`_lake_depth_filter`).
2. **Step 4** — bilinear interpolation of the regular lat/lon field onto
   ``target_grid`` via :func:`pproc.climate.mir_ops.interpolate`.
3. **Step 5** — clamp to ``[0.5, 10000] m`` via the pproc formula
   evaluator (``max(min(lake_depth, 10000), 0.5)``).
4. **Step 6** — encode with ``paramId=228007``, ``typeOfLevel=surface``,
   ``packingType=grid_simple`` via :func:`pproc.common.io.encode_grib`.

Steps 1 (resolution-ladder mapping) and 7 (archival) stay in the ksh
wrapper — the product only knows about GRIB bytes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from conflator import CLIArg
from pydantic import ConfigDict, Field

from pproc.climate import mir_ops
from pproc.climate.generate.config import BaseGenerateConfig
from pproc.climate.generate.products._lake_depth_filter import compute_regridded_depth
from pproc.common.io import decode_grib, encode_grib
from pproc.formula import evaluate_formula

__all__ = [
    "FIELD_NAME",
    "DESCRIPTION",
    "CONFIG",
    "LakeDepthConfig",
    "generate",
]


FIELD_NAME = "lake-depth"
DESCRIPTION = (
    "Lake depth on the target grid via mode-filtering source data at ~30 "
    "arc-second resolution, bilinear regridding, and clamping to [0.5, 10000] m."
)


logger = logging.getLogger(__name__)


class LakeDepthConfig(BaseGenerateConfig):
    """Configuration for lake-depth generation.

    Field-to-ksh-variable mapping (see ``ifs-scripts/clim/generate_lake_depth.ksh``):

    ============================  ==================================================
    Field                         ksh env-var / source
    ============================  ==================================================
    ``world_depth``               ``${CLIMFIELDS_SOURCEDATA}/World_DEPTH.dat``
    ``world_depth_status``        ``${CLIMFIELDS_SOURCEDATA}/World_DEPTHStatus.dat``
    ``depth_template``            ``${CLIMFIELDS_SOURCEDATA}/grib_templates/
                                  RegularLatLonCellCentered_${INCRLL}_${INCRLL}``
    ``resol_km``                  ``$RESOLL`` (block-aggregation factor)
    ``nlat``                      ``$NLATLL``
    ``nlon``                      ``$NLONLL``
    ``target_grid`` (base)        ``$MIR_GTYPE_SET``
    ``lakedl_out``                output path (archived to ``$XDATA_IFS`` by wrapper)
    ============================  ==================================================
    """

    model_config = ConfigDict(extra="forbid")

    world_depth: Annotated[
        Path,
        CLIArg("--world-depth", default=None),
        Field(
            description=(
                "Path to World_DEPTH.dat (43200×21600 float32 lake mean "
                "depths in metres, little-endian, Fortran order)."
            ),
        ),
    ] = Path("./World_DEPTH.dat")

    world_depth_status: Annotated[
        Path,
        CLIArg("--world-depth-status", default=None),
        Field(
            description=(
                "Path to World_DEPTHStatus.dat (43200×21600 int8 source "
                "flags per the GLDBv3/GEBCO/geological/default hierarchy)."
            ),
        ),
    ] = Path("./World_DEPTHStatus.dat")

    depth_template: Annotated[
        Path,
        CLIArg("--depth-template", default=None),
        Field(
            description=(
                "Path to a regular lat/lon GRIB template matching the "
                "``resol_km`` × ``nlat`` × ``nlon`` grid (ksh: "
                "``${CLIMFIELDS_SOURCEDATA}/grib_templates/"
                "RegularLatLonCellCentered_${INCRLL}_${INCRLL}``)."
            ),
        ),
    ]

    resol_km: Annotated[
        int,
        CLIArg("--resol-km", type=int, default=None),
        Field(
            gt=0,
            description=(
                "Block-aggregation factor: number of 1 km source pixels "
                "per regular lat/lon output pixel (ksh: ``$RESOLL``)."
            ),
        ),
    ]

    nlat: Annotated[
        int,
        CLIArg("--nlat", type=int, default=None),
        Field(
            gt=0,
            description=(
                "Number of latitude points in the regridded regular "
                "lat/lon grid (ksh: ``$NLATLL``)."
            ),
        ),
    ]

    nlon: Annotated[
        int,
        CLIArg("--nlon", type=int, default=None),
        Field(
            gt=0,
            description=(
                "Number of longitude points in the regridded regular "
                "lat/lon grid (ksh: ``$NLONLL``)."
            ),
        ),
    ]

    # ``target_grid`` is inherited from BaseGenerateConfig as Optional; the
    # ksh always sets $MIR_GTYPE_SET so we redeclare it as required (same
    # trick sso.py uses).
    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(
            min_length=1,
            description=(
                "Target output grid spec (ksh: ``$MIR_GTYPE_SET``, e.g. 'N320'). "
                "Bilinear interpolation is applied from the regular lat/lon grid "
                "onto this target."
            ),
        ),
    ]

    lakedl_out: Annotated[
        Path,
        CLIArg("--lakedl-out", default=None),
        Field(
            description=(
                "Output path for the final lake-depth GRIB (ksh: "
                "``$fileName`` = ``lakedl``; the wrapper moves this to "
                "``${XDATA_IFS}`` and archives with ecp). Default ``./lakedl``."
            ),
        ),
    ] = Path("./lakedl")


CONFIG = LakeDepthConfig


def generate(config: LakeDepthConfig) -> dict[str, bytes]:
    """Compute the lake-depth field on the target grid.

    See :mod:`pproc.climate.generate.products.lake_depth` module docstring
    for the four-step pipeline.

    Returns
    -------
    dict[str, bytes]
        Single-entry mapping ``{"lakedl": encoded_grib_bytes}``. The CLI
        layer writes this to ``config.lakedl_out``.
    """
    # -- validate inputs exist -----------------------------------------
    for label, path in (
        ("world-depth", config.world_depth),
        ("world-depth-status", config.world_depth_status),
        ("depth-template", config.depth_template),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} input '{path}' does not exist (or is not a regular file)"
            )

    # -- Step 3: mode-filter aggregation -------------------------------
    logger.info(
        "lake-depth stage 3 (mode filter): source=%s status=%s → %dx%d @ %d km",
        config.world_depth,
        config.world_depth_status,
        config.nlon,
        config.nlat,
        config.resol_km,
    )
    depth_1d = compute_regridded_depth(
        config.world_depth,
        config.world_depth_status,
        new_res=config.resol_km,
        nlat_new=config.nlat,
        nlon_new=config.nlon,
    )
    logger.info(
        "lake-depth stage 3 complete: %d points, min=%.3f max=%.3f",
        depth_1d.size,
        float(depth_1d.min()),
        float(depth_1d.max()),
    )

    # Encode on the regular-lat/lon template (equivalent to the ksh's
    # `World_DEPTH_ll.grb`). encode_grib expects float64.
    template_bytes = config.depth_template.read_bytes()
    depth_ll_bytes = encode_grib(depth_1d.astype("float64"), template_bytes)
    logger.info(
        "lake-depth stage 3 encoded onto regular lat/lon template (%d bytes)",
        len(depth_ll_bytes),
    )

    # -- Step 4: bilinear interpolation to target grid -----------------
    logger.info("lake-depth stage 4 (bilinear → %s)", config.target_grid)
    depth_target_bytes = mir_ops.interpolate(
        depth_ll_bytes, grid=config.target_grid, method="structured-bilinear"
    )
    logger.info("lake-depth stage 4 complete (%d bytes)", len(depth_target_bytes))

    # -- Step 5: clamp to [0.5, 10000] ---------------------------------
    logger.info("lake-depth stage 5 (clamp to [0.5, 10000] m)")
    depth_target, _ = decode_grib(depth_target_bytes)
    clamped = evaluate_formula(
        "max(min(lake_depth, 10000), 0.5)",
        {"lake_depth": depth_target},
    )
    logger.info(
        "lake-depth stage 5 complete: min=%.3f max=%.3f",
        float(clamped.min()),
        float(clamped.max()),
    )

    # -- Step 6: final encode with paramId + typeOfLevel + packingType --
    # TODO: check typeOfLevel - is surface correct? ecCodes > 2.16.0 think not, and it should be 'entireLake'
    logger.info("lake-depth stage 6 (encode with paramId=228007, typeOfLevel=surface)")
    metadata: dict = {
        "paramId": 228007,
        "typeOfLevel": "surface",
        "packingType": "grid_simple",
    }
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    encoded = encode_grib(clamped, depth_target_bytes, metadata=metadata)
    logger.info("lake-depth stage 6 complete (%d bytes)", len(encoded))

    return {"lakedl": encoded}
