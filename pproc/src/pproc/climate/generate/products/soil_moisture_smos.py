# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``soil-moisture-smos`` product: SMOS brightness temperature bias correction.

Faithful port of ``ifs-scripts/clim-pproc/generate_soil_moisture_smos.ksh``.

The ksh loops over the two source files ``t511bcXX_monthly`` and
``t511bcYY_monthly``. That loop is over LOGICAL products; the wrapper
keeps the loop. The ksh also runs ``grib_copy -w paramId=172`` on the
source lsmoro file to extract the lsm — that stays in the wrapper too
(eccodes utility). This product just does the ``nearest-lsm``
interpolation and preserves paramIds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import eccodes
from conflator import CLIArg
from pydantic import ConfigDict, Field

from pproc.climate import mir_ops
from pproc.climate.generate.config import BaseGenerateConfig

__all__ = [
    "FIELD_NAME",
    "DESCRIPTION",
    "CONFIG",
    "generate",
    "SoilMoistureSmosConfig",
]


FIELD_NAME = "soil-moisture-smos"
DESCRIPTION = (
    "SMOS brightness temperature bias correction on target grid (nearest-lsm)."
)


logger = logging.getLogger(__name__)


class SoilMoistureSmosConfig(BaseGenerateConfig):
    """Config for the soil-moisture-smos product."""

    model_config = ConfigDict(extra="forbid")

    smos_in: Annotated[
        Path,
        CLIArg("--smos-in", default=None),
        Field(
            description="Source SMOS parameter GRIB (t511bcXX_monthly / t511bcYY_monthly)."
        ),
    ] = Path("./smos_in")

    source_lsm_in: Annotated[
        Path,
        CLIArg("--source-lsm-in", default=None),
        Field(description="Land-sea mask of the SOURCE grid (single-paramId GRIB)."),
    ] = Path("./source_lsm")

    target_lsm_in: Annotated[
        Path,
        CLIArg("--target-lsm-in", default=None),
        Field(description="Land-sea mask on TARGET grid."),
    ] = Path("./land_mask")

    smos_out: Annotated[
        Path,
        CLIArg("--smos-out", default=None),
        Field(description="Output path. Default ``./smos_out``."),
    ] = Path("./smos_out")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = SoilMoistureSmosConfig


def generate(config: SoilMoistureSmosConfig) -> dict[str, bytes]:
    """Nearest-lsm interpolation, preserve paramIds, set packing=grid_simple."""
    logger.info(
        "soil-moisture-smos: %s → %s (nearest-lsm)", config.smos_in, config.target_grid
    )
    src = config.smos_in.read_bytes()
    regridded = mir_ops.interpolate(
        src,
        grid=config.target_grid,
        method="nearest-lsm",
        lsm_selection="file",
        lsm_file_input=str(config.source_lsm_in),
        lsm_file_output=str(config.target_lsm_in),
    )

    out_chunks: list[bytes] = []
    reader = eccodes.MemoryReader(regridded)
    n = 0
    for message in reader:
        msg = message.copy()
        keys: dict = {"packingType": "grid_simple"}
        if config.bits_per_value is not None:
            keys["bitsPerValue"] = config.bits_per_value
        msg.set(keys, check_values=True)
        out_chunks.append(msg.get_buffer())
        n += 1
    logger.info("soil-moisture-smos: %d messages re-packed", n)
    return {"smos": b"".join(out_chunks)}
