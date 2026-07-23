# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``oceanic-emissions`` product: monthly DMS emissions on target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_oceanic_emissions.ksh``.

The ksh loops over 12 monthly ``dms_<mm>_new.grb`` files and appends the
results into a single ``month_dms``. That loop is algorithm-intrinsic and
moves inside this product.

Per month:

1. ``nearest-neighbour`` interpolation onto target grid with area
   ``90/0/-90/360``.
2. Mask out non-ocean: ``field * ocean_mask - 99 * (1 - ocean_mask)``.
3. Encode with ``paramId=210043, dataDate=9999<mm>15,
   packingType=grid_simple``.
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

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "OceanicEmissionsConfig"]


FIELD_NAME = "oceanic-emissions"
DESCRIPTION = (
    "Monthly DMS oceanic emissions on target grid (nearest-neighbour + ocean mask)."
)


logger = logging.getLogger(__name__)


class OceanicEmissionsConfig(BaseGenerateConfig):
    """Config for the oceanic-emissions product."""

    model_config = ConfigDict(extra="forbid")

    dms_in_prefix: Annotated[
        Path,
        CLIArg("--dms-in-prefix", default=None),
        Field(
            description=(
                "Prefix for monthly DMS source files; each month ``mm`` reads "
                "``<prefix>_<mm>.grb`` (ksh: ``${CLIMFIELDS_SOURCEDATA}/dms``)."
            ),
        ),
    ] = Path("./dms")

    ocean_mask_in: Annotated[
        Path,
        CLIArg("--ocean-mask-in", default=None),
        Field(description="Ocean mask GRIB on target grid."),
    ] = Path("./ocean_mask")

    month_dms_out: Annotated[
        Path,
        CLIArg("--month-dms-out", default=None),
        Field(description="Output path. Default ``./month_dms``."),
    ] = Path("./month_dms")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = OceanicEmissionsConfig


def generate(config: OceanicEmissionsConfig) -> dict[str, bytes]:
    """Interpolate + ocean-mask each of 12 monthly DMS files."""
    logger.info(
        "oceanic-emissions: prefix=%s → %s", config.dms_in_prefix, config.target_grid
    )
    ocean_mask, _ = decode_grib(config.ocean_mask_in.read_bytes())

    out_chunks: list[bytes] = []
    for month in range(1, 13):
        mm = f"{month:02d}"
        in_path = config.dms_in_prefix.with_name(
            f"{config.dms_in_prefix.name}_{mm}.grb"
        )
        src = in_path.read_bytes()
        regridded = mir_ops.interpolate(
            src,
            grid=config.target_grid,
            method="nearest-neighbour",
            area="90/0/-90/360",
        )
        field, _ = decode_grib(regridded)
        masked = evaluate_formula(
            "field * ocean_mask - 99 * (1 - ocean_mask)",
            {"field": field, "ocean_mask": ocean_mask},
        )

        metadata: dict = {
            "paramId": 210043,
            "dataDate": int(f"9999{mm}15"),
            "packingType": "grid_simple",
        }
        if config.bits_per_value is not None:
            metadata["bitsPerValue"] = config.bits_per_value

        out_chunks.append(encode_grib(masked, regridded, metadata=metadata))
        logger.info("oceanic-emissions: month %s done", mm)

    return {"month_dms": b"".join(out_chunks)}
