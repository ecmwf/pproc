# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``urban-cover`` product: interpolate urban cover, apply land mask.

Faithful port of ``ifs-scripts/clim-pproc/generate_urban_cover.ksh``.

Two stages per invocation:

1. Regrid the source urban cover onto the target grid (grid-box-average).
2. Multiply by the land mask (``field * land_mask``).

The wrapper's outer loop over time-slices (``urban``, ``urban_1975.grb``,
..., ``urban_2005.grb``) is a loop over LOGICAL products and stays in
the wrapper; each iteration calls this product once with a fresh
``--urban-cover-in`` / ``--urban-cover-out`` pair and a ``--date`` value.
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

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "UrbanCoverConfig"]


FIELD_NAME = "urban-cover"
DESCRIPTION = "Urban cover on target grid (grid-box-average + land-mask)."


logger = logging.getLogger(__name__)


class UrbanCoverConfig(BaseGenerateConfig):
    """Config for the urban-cover product."""

    model_config = ConfigDict(extra="forbid")

    urban_cover_in: Annotated[
        Path,
        CLIArg("--urban-cover-in", default=None),
        Field(description="Source urban-cover GRIB."),
    ] = Path("./urban")

    land_mask_in: Annotated[
        Path,
        CLIArg("--land-mask-in", default=None),
        Field(description="Land-mask GRIB on target grid."),
    ] = Path("./land_mask")

    urban_cover_out: Annotated[
        Path,
        CLIArg("--urban-cover-out", default=None),
        Field(description="Output path. Default ``./urban``."),
    ] = Path("./urban")

    date: Annotated[
        int,
        CLIArg("--date", type=int, default=None),
        Field(
            description=(
                "GRIB ``date`` key value (ksh: ``99990615`` — a climatological "
                "mid-year sentinel; time-slice variants override this per year)."
            ),
        ),
    ] = 99990615

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = UrbanCoverConfig


def generate(config: UrbanCoverConfig) -> dict[str, bytes]:
    """Regrid, mask, and set paramId/date/packing."""
    logger.info(
        "urban-cover: %s + %s → %s (date=%d)",
        config.urban_cover_in,
        config.land_mask_in,
        config.target_grid,
        config.date,
    )
    src = config.urban_cover_in.read_bytes()
    regridded = mir_ops.interpolate(
        src, grid=config.target_grid, method="grid-box-average"
    )
    field, _ = decode_grib(regridded)
    land_mask, _ = decode_grib(config.land_mask_in.read_bytes())
    masked = evaluate_formula(
        "field * land_mask",
        {"field": field, "land_mask": land_mask},
    )

    metadata: dict = {
        "paramId": 229001,
        "date": config.date,
        "packingType": "grid_simple",
        "bitsPerValue": 16,  # ksh: setBitsPerValue=16 (eccodes accessor → bpv)
    }
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    encoded = encode_grib(masked, regridded, metadata=metadata)
    return {"urban_cover": encoded}
