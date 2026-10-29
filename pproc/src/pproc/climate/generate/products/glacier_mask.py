# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``glacier-mask`` product: derive glacier mask and glacier-free land mask.

Faithful port of ``ifs-scripts/clim-pproc/generate_glacier_mask.ksh``.

Pipeline (three formulae, two logical outputs):

1. ``glacier_mask_raw = (glacier_cover > 0.5) * 1 + 0``
2. ``glacier_mask     = glacier_mask_raw * land_mask``  (consistency check)
3. ``glacier_free_land_mask = land_mask - glacier_mask``

No interpolation; inputs are expected on the target grid.
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

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "GlacierMaskConfig"]


FIELD_NAME = "glacier-mask"
DESCRIPTION = "Binary glacier and glacier-free land masks (2 outputs)."


logger = logging.getLogger(__name__)


class GlacierMaskConfig(BaseGenerateConfig):
    """Config for the glacier-mask product."""

    model_config = ConfigDict(extra="forbid")

    glacier_cover_in: Annotated[
        Path,
        CLIArg("--glacier-cover-in", default=None),
        Field(description="Input glacier-cover GRIB (``cicecap`` on target grid)."),
    ] = Path("./cicecap")

    land_mask_in: Annotated[
        Path,
        CLIArg("--land-mask-in", default=None),
        Field(description="Input land-mask GRIB on target grid."),
    ] = Path("./land_mask")

    glacier_mask_out: Annotated[
        Path,
        CLIArg("--glacier-mask-out", default=None),
        Field(description="Output glacier-mask GRIB. Default ``./glacier_mask``."),
    ] = Path("./glacier_mask")

    glacier_free_land_mask_out: Annotated[
        Path,
        CLIArg("--glacier-free-land-mask-out", default=None),
        Field(
            description=(
                "Output glacier-free land-mask GRIB. Default ``./glacier_free_land_mask``."
            ),
        ),
    ] = Path("./glacier_free_land_mask")


CONFIG = GlacierMaskConfig


def generate(config: GlacierMaskConfig) -> dict[str, bytes]:
    """Compute both glacier_mask and glacier_free_land_mask."""
    glacier_bytes = config.glacier_cover_in.read_bytes()
    land_bytes = config.land_mask_in.read_bytes()
    glacier_cover, _ = decode_grib(glacier_bytes)
    land_mask, _ = decode_grib(land_bytes)

    # Step 1: raw threshold
    glacier_mask_raw = evaluate_formula(
        "(glacier_cover > 0.5) * 1 + 0",
        {"glacier_cover": glacier_cover},
    )
    # Step 2: consistency with land
    glacier_mask = evaluate_formula(
        "glacier_mask * land_mask",
        {"glacier_mask": glacier_mask_raw, "land_mask": land_mask},
    )
    # Step 3: glacier-free land
    glacier_free_land = evaluate_formula(
        "land_mask - glacier_mask",
        {"land_mask": land_mask, "glacier_mask": glacier_mask},
    )
    logger.info(
        "glacier-mask: glacier=%d, glacier_free_land=%d, land=%d",
        int((glacier_mask > 0).sum()),
        int((glacier_free_land > 0).sum()),
        int((land_mask > 0).sum()),
    )

    metadata: dict = {"packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    # Templates: glacier_mask inherits identity from the cicecap input (as the
    # original ksh does — first formula writes over ``mir_output`` which
    # started from ``$inFile`` = cicecap). glacier_free_land_mask inherits
    # identity from the land mask (the original ksh's mir_output is written
    # from cat $maskFile $fileName1, positional-first is land_mask so the GRIB
    # header comes from there).
    return {
        "glacier_mask": encode_grib(glacier_mask, glacier_bytes, metadata=metadata),
        "glacier_free_land_mask": encode_grib(
            glacier_free_land, land_bytes, metadata=metadata
        ),
    }
