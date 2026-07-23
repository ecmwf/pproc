# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``albedo-single-stream`` product: monthly single-stream albedo on target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_albedo_single_stream.ksh``.
The source ``month_alb`` file carries 12 monthly messages; mir preserves the
multi-message layout across the ``grid-box-average`` interpolation, so we
iterate the output messages here and apply metadata to each one.
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
from pproc.common.io import encode_grib

__all__ = [
    "FIELD_NAME",
    "DESCRIPTION",
    "CONFIG",
    "generate",
    "AlbedoSingleStreamConfig",
]


FIELD_NAME = "albedo-single-stream"
DESCRIPTION = "Monthly single-stream albedo on target grid (grid-box-average)."


logger = logging.getLogger(__name__)


class AlbedoSingleStreamConfig(BaseGenerateConfig):
    """Config for the albedo-single-stream product."""

    model_config = ConfigDict(extra="forbid")

    month_alb_in: Annotated[
        Path,
        CLIArg("--month-alb-in", default=None),
        Field(description="Source monthly single-stream albedo GRIB (12 messages)."),
    ] = Path("./month_alb")

    month_alb_out: Annotated[
        Path,
        CLIArg("--month-alb-out", default=None),
        Field(description="Output path. Default ``./month_alb``."),
    ] = Path("./month_alb")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = AlbedoSingleStreamConfig


def generate(config: AlbedoSingleStreamConfig) -> dict[str, bytes]:
    """Regrid + set paramId=174 on each of the (12 monthly) messages."""
    logger.info(
        "albedo-single-stream: %s → %s", config.month_alb_in, config.target_grid
    )
    src = config.month_alb_in.read_bytes()
    regridded = mir_ops.interpolate(
        src, grid=config.target_grid, method="grid-box-average"
    )

    metadata: dict = {"paramId": 174, "packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    # Iterate over each monthly message, re-encode with metadata, concat.
    out_chunks: list[bytes] = []
    reader = eccodes.MemoryReader(regridded)
    n = 0
    for message in reader:
        wire = message.get_buffer()
        values = message.get_array("values")
        out_chunks.append(encode_grib(values, wire, metadata=metadata))
        n += 1
    logger.info("albedo-single-stream: re-encoded %d monthly messages", n)

    return {"month_alb": b"".join(out_chunks)}
