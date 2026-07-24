# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``ocean-mask`` product: derive an ocean mask from land + lake masks.

Faithful port of ``ifs-scripts/clim-pproc/generate_ocean_mask.ksh``. The ksh
runs ``mir_compute --formula "1 - land_mask - lake_mask"`` then
``grib_set -s packingType=grid_simple``; the pproc port evaluates the same
expression via :func:`pproc.formula.evaluate_formula`. Both steps are folded
into this product. No interpolation.
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

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "OceanMaskConfig"]


FIELD_NAME = "ocean-mask"
DESCRIPTION = "Binary ocean mask = 1 - land_mask - lake_mask."


logger = logging.getLogger(__name__)


class OceanMaskConfig(BaseGenerateConfig):
    """Config for the ocean-mask product."""

    model_config = ConfigDict(extra="forbid")

    land_mask_in: Annotated[
        Path,
        CLIArg("--land-mask-in", default=None),
        Field(
            description="Path to input land-mask GRIB (ksh: ``${XDATA_IFS}/land_mask``)."
        ),
    ] = Path("./land_mask")

    lake_mask_in: Annotated[
        Path,
        CLIArg("--lake-mask-in", default=None),
        Field(
            description="Path to input lake-mask GRIB (ksh: ``${XDATA_IFS}/lake_mask``)."
        ),
    ] = Path("./lake_mask")

    ocean_mask_out: Annotated[
        Path,
        CLIArg("--ocean-mask-out", default=None),
        Field(description="Output binary ocean-mask GRIB. Default ``./ocean_mask``."),
    ] = Path("./ocean_mask")


CONFIG = OceanMaskConfig


def generate(config: OceanMaskConfig) -> dict[str, bytes]:
    """Compute ocean_mask = 1 - land_mask - lake_mask."""
    land_bytes = config.land_mask_in.read_bytes()
    lake_bytes = config.lake_mask_in.read_bytes()
    land, _ = decode_grib(land_bytes)
    lake, _ = decode_grib(lake_bytes)

    ocean = evaluate_formula(
        "1 - land_mask - lake_mask",
        {"land_mask": land, "lake_mask": lake},
    )
    logger.info(
        "ocean-mask: %d ocean pixels of %d", int((ocean > 0).sum()), int(ocean.size)
    )

    metadata: dict = {"packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    # Use the land_mask as template (arbitrary but stable: they have the same grid).
    encoded = encode_grib(ocean, land_bytes, metadata=metadata)
    return {"ocean_mask": encoded}
