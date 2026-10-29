# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``land-cover`` product: conservatively interpolate 30" land cover to target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_land_cover.ksh``.
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

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "LandCoverConfig"]


FIELD_NAME = "land-cover"
DESCRIPTION = "Land cover (lsm) on target grid (grid-box-average interpolation)."


logger = logging.getLogger(__name__)


class LandCoverConfig(BaseGenerateConfig):
    """Config for the land-cover product."""

    model_config = ConfigDict(extra="forbid")

    lsm_in: Annotated[
        Path,
        CLIArg("--lsm-in", default=None),
        Field(
            description="Source land-cover GRIB (ksh: ``${CLIMFIELDS_SOURCEDATA}/lsm``)."
        ),
    ] = Path("./lsm")

    lsm_out: Annotated[
        Path,
        CLIArg("--lsm-out", default=None),
        Field(description="Output land-cover GRIB path. Default ``./lsm``."),
    ] = Path("./lsm")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = LandCoverConfig


def generate(config: LandCoverConfig) -> dict[str, bytes]:
    """Interpolate source lsm via grid-box-average and set metadata."""
    logger.info("land-cover: %s → %s", config.lsm_in, config.target_grid)
    src = config.lsm_in.read_bytes()
    regridded = mir_ops.interpolate(
        src, grid=config.target_grid, method="grid-box-average"
    )
    values, _ = decode_grib(regridded)

    metadata: dict = {
        "shortName": "lsm",
        "dataType": "an",
        "setLocalDefinition": 1,
        "localDefinitionNumber": 1,
        "packingType": "grid_simple",
    }
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    encoded = encode_grib(values, regridded, metadata=metadata)
    return {"lsm": encoded}
