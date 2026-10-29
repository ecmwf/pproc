# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``subgrid-orography-sdfor`` product: 1-5 km orographic std-dev on target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_subgrid_orography_sdfor.ksh``.

Two paths:

* If the ``orog-variance-1km-5km`` GRIB exists → skip Stage 1 and just
  regrid + sqrt + mask.
* If not → compute the 1-5 km variance from the 30" orography via the
  identity ``sigma^2 = mean(x^2) - (mean(x))^2`` on N2000, then continue.

May exhibit the same float32/float64 D-F1 drift as SSO for the sqrt-of-
aggregated-variance step; if the harness reports a diff, revisit and
consider the atol pattern the SSO product uses.
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

__all__ = [
    "FIELD_NAME",
    "DESCRIPTION",
    "CONFIG",
    "generate",
    "SubgridOrographySdforConfig",
]


FIELD_NAME = "subgrid-orography-sdfor"
DESCRIPTION = "Std dev of orography 1-5km on target grid (sdfor)."


logger = logging.getLogger(__name__)


class SubgridOrographySdforConfig(BaseGenerateConfig):
    """Config for the subgrid-orography-sdfor product."""

    model_config = ConfigDict(extra="forbid")

    orog_variance_1km_5km_in: Annotated[
        Path,
        CLIArg("--orog-variance-1km-5km-in", default=None),
        Field(
            description=(
                "Pre-computed orography variance on N2000. If missing, will be "
                "built from ``--orog-in`` via the mean-square identity."
            ),
        ),
    ] = Path("./orog_variance_1km-5km")

    orog_variance_1km_5km_out: Annotated[
        Path,
        CLIArg("--orog-variance-1km-5km-out", default=None),
        Field(
            description=(
                "If the variance had to be built here (Stage 1 fired), it is "
                "cached to this path. Otherwise ignored."
            ),
        ),
    ] = Path("./orog_variance_1km-5km")

    orog_in: Annotated[
        Path,
        CLIArg("--orog-in", default=None),
        Field(
            description='Source 30" orography (used only when variance file missing).'
        ),
    ] = Path("./orog")

    land_mask_in: Annotated[
        Path,
        CLIArg("--land-mask-in", default=None),
        Field(description="Land mask on target grid."),
    ] = Path("./land_mask")

    sdfor_out: Annotated[
        Path,
        CLIArg("--sdfor-out", default=None),
        Field(description="Output path. Default ``./sdfor``."),
    ] = Path("./sdfor")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = SubgridOrographySdforConfig


def generate(config: SubgridOrographySdforConfig) -> dict[str, bytes]:
    """Compute sdfor via total-variance identity, then sqrt + mask."""
    logger.info("subgrid-orography-sdfor: → %s", config.target_grid)

    outputs: dict[str, bytes] = {}

    # ---- Stage 1: get the 1-5 km variance ---------------------------------
    if config.orog_variance_1km_5km_in.is_file():
        variance_bytes = config.orog_variance_1km_5km_in.read_bytes()
        logger.info("sdfor stage 1: using pre-computed variance")
    else:
        logger.info("sdfor stage 1: building variance from orog on N2000")
        orog_src = config.orog_in.read_bytes()

        # mean(x) on N2000
        orog_n2000_bytes = mir_ops.interpolate(
            orog_src, grid="N2000", method="grid-box-average"
        )
        orog_n2000, _ = decode_grib(orog_n2000_bytes)

        # x^2 on source grid
        orog_vals, _ = decode_grib(orog_src)
        orog_sq = evaluate_formula("field * field", {"field": orog_vals})
        orog_sq_bytes = encode_grib(orog_sq, orog_src)

        # mean(x^2) on N2000
        orog_sq_n2000_bytes = mir_ops.interpolate(
            orog_sq_bytes, grid="N2000", method="grid-box-average"
        )
        orog_sq_n2000, _ = decode_grib(orog_sq_n2000_bytes)

        variance_vals = evaluate_formula(
            "abs( mean_square - (mean * mean) )",
            {"mean_square": orog_sq_n2000, "mean": orog_n2000},
        )
        variance_bytes = encode_grib(variance_vals, orog_sq_n2000_bytes)
        outputs["orog_variance_1km_5km"] = variance_bytes

    # ---- Stage 2: regrid variance to target ------------------------------
    variance_target_bytes = mir_ops.interpolate(
        variance_bytes, grid=config.target_grid, method="grid-box-average"
    )
    variance_target, _ = decode_grib(variance_target_bytes)

    # ---- Stage 3: sqrt --------------------------------------------------
    std_target = evaluate_formula("sqrt(field)", {"field": variance_target})

    # ---- Stage 4: land-mask ---------------------------------------------
    land_mask, _ = decode_grib(config.land_mask_in.read_bytes())
    masked = evaluate_formula(
        "field * land_mask", {"field": std_target, "land_mask": land_mask}
    )

    metadata: dict = {"shortName": "sdfor", "packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    outputs["sdfor"] = encode_grib(masked, variance_target_bytes, metadata=metadata)
    logger.info("subgrid-orography-sdfor: complete")
    return outputs
