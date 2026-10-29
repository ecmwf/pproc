# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``wetland-cover`` product: monthly wetland cover on target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_wetland_cover.ksh``.

The ksh loops over the 12 monthly files ``wetlandf_${mm}.grb`` and appends
the results into a single ``wetlandf`` file. That loop is a component of
ONE product's algorithm (the output is a single logical wetland file with
12 monthly messages), so it moves INSIDE this product.

Per month:

1. Interpolate ``wetlandf_${mm}.grb`` with ``grid-box-average`` and area
   ``90/0/-90/360``.
2. Apply ``field * land_mask``.
3. Encode with ``paramId=229007, date=9999<mm>15, setBitsPerValue=16,
   packingType=grid_simple``.

The 12 monthly messages are then concatenated into the single output.
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

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "WetlandCoverConfig"]


FIELD_NAME = "wetland-cover"
DESCRIPTION = "Monthly wetland cover on target grid (12-month loop + land mask)."


logger = logging.getLogger(__name__)


class WetlandCoverConfig(BaseGenerateConfig):
    """Config for the wetland-cover product."""

    model_config = ConfigDict(extra="forbid")

    wetlandf_in_prefix: Annotated[
        Path,
        CLIArg("--wetlandf-in-prefix", default=None),
        Field(
            description=(
                "Path prefix for monthly wetland-cover files. Each month "
                "``mm in {01..12}`` reads ``<prefix>_<mm>.grb``. ksh: "
                "``${CLIMFIELDS_SOURCEDATA}/wetlandf``."
            ),
        ),
    ] = Path("./wetlandf")

    land_mask_in: Annotated[
        Path,
        CLIArg("--land-mask-in", default=None),
        Field(description="Land mask on target grid."),
    ] = Path("./land_mask")

    wetlandf_out: Annotated[
        Path,
        CLIArg("--wetlandf-out", default=None),
        Field(
            description="Output GRIB path (12-message monthly). Default ``./wetlandf``."
        ),
    ] = Path("./wetlandf")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = WetlandCoverConfig


def generate(config: WetlandCoverConfig) -> dict[str, bytes]:
    """Interpolate + mask each of 12 monthly wetland files, return concatenated bytes."""
    logger.info(
        "wetland-cover: prefix=%s + mask=%s → %s",
        config.wetlandf_in_prefix,
        config.land_mask_in,
        config.target_grid,
    )
    land_mask, _ = decode_grib(config.land_mask_in.read_bytes())

    out_chunks: list[bytes] = []
    for month in range(1, 13):
        mm = f"{month:02d}"
        in_path = config.wetlandf_in_prefix.with_name(
            f"{config.wetlandf_in_prefix.name}_{mm}.grb"
        )
        src = in_path.read_bytes()
        regridded = mir_ops.interpolate(
            src,
            grid=config.target_grid,
            method="grid-box-average",
            area="90/0/-90/360",
        )
        field, _ = decode_grib(regridded)
        masked = evaluate_formula(
            "field * land_mask",
            {"field": field, "land_mask": land_mask},
        )

        metadata: dict = {
            "paramId": 229007,
            "date": int(f"9999{mm}15"),
            "packingType": "grid_simple",
            "bitsPerValue": 16,  # ksh: setBitsPerValue=16
        }
        if config.bits_per_value is not None:
            metadata["bitsPerValue"] = config.bits_per_value

        out_chunks.append(encode_grib(masked, regridded, metadata=metadata))
        logger.info("wetland-cover: month %s done", mm)

    return {"wetlandf": b"".join(out_chunks)}
