# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``lake-cover`` product: conservatively interpolate lake/waterbody cover to target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_lake_cover.ksh``. The ksh
loops over two source files (``clake``, ``waterbody_c2``); that loop is a
loop over LOGICAL products (each yields its own output) and stays in the
wrapper, which calls this tool once per file.
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

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "LakeCoverConfig"]


FIELD_NAME = "lake-cover"
DESCRIPTION = (
    "Inland water / lake cover on target grid (grid-box-average interpolation)."
)


logger = logging.getLogger(__name__)


class LakeCoverConfig(BaseGenerateConfig):
    """Config for the lake-cover product."""

    model_config = ConfigDict(extra="forbid")

    lake_cover_in: Annotated[
        Path,
        CLIArg("--lake-cover-in", default=None),
        Field(
            description=(
                "Source lake-cover GRIB (ksh: ``${CLIMFIELDS_SOURCEDATA}/clake`` "
                "or ``waterbody_c2``)."
            ),
        ),
    ] = Path("./clake")

    lake_cover_out: Annotated[
        Path,
        CLIArg("--lake-cover-out", default=None),
        Field(description="Output lake-cover GRIB path. Default ``./clake``."),
    ] = Path("./clake")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = LakeCoverConfig


def generate(config: LakeCoverConfig) -> dict[str, bytes]:
    """Interpolate lake cover and set metadata."""
    logger.info("lake-cover: %s → %s", config.lake_cover_in, config.target_grid)
    src = config.lake_cover_in.read_bytes()
    regridded = mir_ops.interpolate(
        src, grid=config.target_grid, method="grid-box-average"
    )
    values, _ = decode_grib(regridded)

    metadata: dict = {"shortName": "cl", "packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    encoded = encode_grib(values, regridded, metadata=metadata)
    return {"lake_cover": encoded}
