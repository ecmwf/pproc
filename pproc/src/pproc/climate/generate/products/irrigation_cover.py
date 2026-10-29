# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``irrigation-cover`` product: conservatively interpolate irrigation cover to target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_irrigation_cover.ksh``.
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

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "IrrigationCoverConfig"]


FIELD_NAME = "irrigation-cover"
DESCRIPTION = "Irrigation cover on target grid (grid-box-average interpolation)."


logger = logging.getLogger(__name__)


class IrrigationCoverConfig(BaseGenerateConfig):
    """Config for the irrigation-cover product."""

    model_config = ConfigDict(extra="forbid")

    irrigation_cover_in: Annotated[
        Path,
        CLIArg("--irrigation-cover-in", default=None),
        Field(description="Source irrigation-cover GRIB."),
    ] = Path("./irrigation_cover")

    irrigation_cover_out: Annotated[
        Path,
        CLIArg("--irrigation-cover-out", default=None),
        Field(description="Output path. Default ``./irrigation_cover``."),
    ] = Path("./irrigation_cover")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = IrrigationCoverConfig


def generate(config: IrrigationCoverConfig) -> dict[str, bytes]:
    logger.info(
        "irrigation-cover: %s → %s", config.irrigation_cover_in, config.target_grid
    )
    src = config.irrigation_cover_in.read_bytes()
    regridded = mir_ops.interpolate(
        src, grid=config.target_grid, method="grid-box-average"
    )
    values, _ = decode_grib(regridded)

    metadata: dict = {"paramId": 228250, "packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    encoded = encode_grib(values, regridded, metadata=metadata)
    return {"irrigation_cover": encoded}
