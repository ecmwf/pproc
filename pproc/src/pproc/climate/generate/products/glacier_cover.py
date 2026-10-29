# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``glacier-cover`` product: conservatively interpolate glacier cover to target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_glacier_cover.ksh``.
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

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "GlacierCoverConfig"]


FIELD_NAME = "glacier-cover"
DESCRIPTION = "Glacier cover (cicecap) on target grid (grid-box-average interpolation)."


logger = logging.getLogger(__name__)


class GlacierCoverConfig(BaseGenerateConfig):
    """Config for the glacier-cover product."""

    model_config = ConfigDict(extra="forbid")

    cicecap_in: Annotated[
        Path,
        CLIArg("--cicecap-in", default=None),
        Field(
            description="Source glacier-cover GRIB (ksh: ``${CLIMFIELDS_SOURCEDATA}/cicecap``)."
        ),
    ] = Path("./cicecap")

    cicecap_out: Annotated[
        Path,
        CLIArg("--cicecap-out", default=None),
        Field(description="Output GRIB path. Default ``./cicecap``."),
    ] = Path("./cicecap")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = GlacierCoverConfig


def generate(config: GlacierCoverConfig) -> dict[str, bytes]:
    logger.info("glacier-cover: %s → %s", config.cicecap_in, config.target_grid)
    src = config.cicecap_in.read_bytes()
    regridded = mir_ops.interpolate(
        src, grid=config.target_grid, method="grid-box-average"
    )
    values, _ = decode_grib(regridded)

    metadata: dict = {"paramId": 207, "packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    encoded = encode_grib(values, regridded, metadata=metadata)
    return {"cicecap": encoded}
