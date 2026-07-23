# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``soil-type`` product: soil type on target grid from 5' data.

Faithful port of ``ifs-scripts/clim-pproc/generate_soil_type.ksh``.

Two interpolation modes selected by target resolution:

* target higher than source (roughly ``RESOL >= 1279 && GTYPE == _4`` or
  ``RESOL > 2047 && GTYPE == l_2``): ``nearest-neighbour``.
* target lower than source: ``grid-box-statistics
  --interpolation-statistics=mode-integral``.

The wrapper decides which branch to take via a ``--interp-method`` flag
(either ``nearest-neighbour`` or ``mode-integral``) so the branching logic
stays as-is in the ksh (it's operational context, not algorithm structure).

Formulae (unchanged from the ksh):

1. ``field * land_mask`` — mask out non-land points.
2. ``field - (field == 9999) * 9998`` — replace 9999-missing values over
   land with soil type 1. **Note**: original mir-compute used the ``=``
   comparison; rewritten to ``==`` for pproc-formula.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from conflator import CLIArg
from pydantic import ConfigDict, Field

from pproc.climate import mir_ops
from pproc.climate.generate.config import BaseGenerateConfig
from pproc.common.io import decode_grib, encode_grib
from pproc.formula import evaluate_formula

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "SoilTypeConfig"]


FIELD_NAME = "soil-type"
DESCRIPTION = "Soil type on target grid (nearest-neighbour or mode-integral)."


logger = logging.getLogger(__name__)


class SoilTypeConfig(BaseGenerateConfig):
    """Config for the soil-type product."""

    model_config = ConfigDict(extra="forbid")

    soil_type_in: Annotated[
        Path,
        CLIArg("--soil-type-in", default=None),
        Field(description="Source 5' soil-type GRIB (``soiltype_10km.grb``)."),
    ] = Path("./soiltype_10km.grb")

    land_mask_in: Annotated[
        Path,
        CLIArg("--land-mask-in", default=None),
        Field(description="Land mask on target grid."),
    ] = Path("./land_mask")

    soil_type_out: Annotated[
        Path,
        CLIArg("--soil-type-out", default=None),
        Field(description="Output path. Default ``./slt``."),
    ] = Path("./slt")

    interp_method: Annotated[
        str,
        CLIArg("--interp-method", default=None),
        Field(
            description=(
                "Interpolation method: ``nearest-neighbour`` (target grid finer "
                "than source) or ``mode-integral`` (target grid coarser than "
                "source; uses ``grid-box-statistics`` with ``interpolation-statistics"
                "=mode-integral``)."
            ),
        ),
    ] = "mode-integral"

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = SoilTypeConfig


def generate(config: SoilTypeConfig) -> dict[str, bytes]:
    """Interpolate, mask, and replace 9999-missing with 1."""
    logger.info(
        "soil-type: %s → %s (%s)",
        config.soil_type_in,
        config.target_grid,
        config.interp_method,
    )
    src = config.soil_type_in.read_bytes()
    if config.interp_method == "nearest-neighbour":
        regridded = mir_ops.interpolate(
            src, grid=config.target_grid, method="nearest-neighbour"
        )
    elif config.interp_method == "mode-integral":
        regridded = mir_ops.interpolate(
            src,
            grid=config.target_grid,
            method="grid-box-statistics",
            interpolation_statistics="mode-integral",
        )
    else:
        raise ValueError(
            f"unsupported --interp-method {config.interp_method!r}; "
            f"expected 'nearest-neighbour' or 'mode-integral'"
        )

    field, _ = decode_grib(regridded)
    land_mask, _ = decode_grib(config.land_mask_in.read_bytes())

    # Formula 1: field * land_mask (original variable order matches ksh)
    masked = evaluate_formula(
        "field * land_mask", {"field": field, "land_mask": land_mask}
    )

    # Formula 2: field - (field == 9999) * 9998
    # note: '=' equality rewritten to '==' for pproc-formula
    fixed = evaluate_formula("field - (field == 9999) * 9998", {"field": masked})
    logger.info(
        "soil-type: %d masked pixels; %d were 9999-fixed",
        int((masked > 0).sum()),
        int((masked == 9999).sum()),
    )

    metadata: dict = {"paramId": 43, "packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    encoded = encode_grib(fixed, regridded, metadata=metadata)
    return {"soil_type": encoded}
