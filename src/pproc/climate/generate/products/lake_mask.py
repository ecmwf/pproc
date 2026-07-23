# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``lake-mask`` product: threshold lake-cover to a binary lake mask.

Faithful port of ``ifs-scripts/clim-pproc/generate_lake_mask.ksh``. The ksh
runs ``pproc-formula`` with the threshold ``(lake_cover >= 0.5) * 1 + 0``,
then ``grib_set -s packingType=grid_simple``. No interpolation is done; the
input is expected to already be on the target grid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from conflator import CLIArg
from pydantic import ConfigDict, Field

from pproc.climate.generate.config import BaseGenerateConfig
from pproc.common.io import decode_grib, encode_grib
from pproc.formula import evaluate_formula

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "LakeMaskConfig"]


FIELD_NAME = "lake-mask"
DESCRIPTION = "Binary lake mask from lake-cover input via threshold >= 0.5."


logger = logging.getLogger(__name__)


class LakeMaskConfig(BaseGenerateConfig):
    """Config for the lake-mask product."""

    model_config = ConfigDict(extra="forbid")

    lake_cover_in: Annotated[
        Path,
        CLIArg("--lake-cover-in", default=None),
        Field(description="Path to input lake-cover GRIB (ksh: ``clake``)."),
    ] = Path("./clake")

    lake_mask_out: Annotated[
        Path,
        CLIArg("--lake-mask-out", default=None),
        Field(description="Output binary lake-mask GRIB. Default ``./lake_mask``."),
    ] = Path("./lake_mask")


CONFIG = LakeMaskConfig


def generate(config: LakeMaskConfig) -> dict[str, bytes]:
    """Threshold lake cover to a binary mask (>= 0.5)."""
    grib_bytes = config.lake_cover_in.read_bytes()
    lake_cover, _ = decode_grib(grib_bytes)

    mask = evaluate_formula(
        "(lake_cover >= 0.5) * 1 + 0",
        {"lake_cover": lake_cover},
    )
    logger.info(
        "lake-mask: %d/%d pixels above threshold",
        int((mask > 0).sum()),
        int(mask.size),
    )

    metadata: dict = {"packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    encoded = encode_grib(mask, grib_bytes, metadata=metadata)
    return {"lake_mask": encoded}
