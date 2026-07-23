# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``land-mask`` product: threshold land-cover to a binary land mask.

Faithful port of ``ifs-scripts/clim-pproc/generate_land_mask.ksh``. The ksh
runs ``pproc-formula --variables land_cover --formula "(land_cover > 0.5) *
1 + 0"`` and then ``grib_set -s packingType=grid_simple``; here we do both
in one shot, in-memory, with metadata applied inside ``generate()``.

No interpolation: the input land cover is expected to already be on the
target grid (the ksh doesn't regrid either — it only thresholds and repacks).
Consequently ``target_grid`` is unused by this product.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from conflator import CLIArg
from pydantic import Field

from pproc.climate.generate.config import BaseGenerateConfig
from pproc.common.io import decode_grib, encode_grib
from pproc.formula import evaluate_formula

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "LandMaskConfig"]


FIELD_NAME = "land-mask"
DESCRIPTION = "Binary land mask from land-cover input via threshold > 0.5."


logger = logging.getLogger(__name__)


class LandMaskConfig(BaseGenerateConfig):
    """Config for the land-mask product.

    One input, one output; no interpolation, no grid config needed.
    """

    land_cover_in: Annotated[
        Path,
        CLIArg("--land-cover-in", default=None),
        Field(
            description=(
                "Path to input land-cover GRIB (ksh: ``$inFile`` — typically "
                "the ``lsm`` file on the target grid)."
            ),
        ),
    ] = Path("./lsm")

    land_mask_out: Annotated[
        Path,
        CLIArg("--land-mask-out", default=None),
        Field(
            description=(
                "Path for the output binary land mask GRIB (ksh: "
                "``$fileName`` = ``land_mask``). Default ``./land_mask``."
            ),
        ),
    ] = Path("./land_mask")


CONFIG = LandMaskConfig


def generate(config: LandMaskConfig) -> dict[str, bytes]:
    """Compute the binary land mask from a land-cover input.

    Steps (mirror the ksh script):

    1. Read the land-cover GRIB from ``config.land_cover_in``.
    2. Threshold ``(land_cover > 0.5) * 1 + 0`` via the pproc formula
       evaluator — identical to what the ksh's ``pproc-formula`` call
       does, but in-memory.
    3. Re-encode against the input as template, setting
       ``packingType=grid_simple`` (equivalent to the ksh's ``grib_set
       -s packingType=grid_simple``) plus optional ``bitsPerValue`` if
       the caller supplied ``--bits-per-value``.

    Returns
    -------
    dict[str, bytes]
        Single-entry mapping ``{"land_mask": encoded_grib_bytes}``. The
        CLI layer writes this to ``config.land_mask_out``.
    """
    grib_bytes = config.land_cover_in.read_bytes()
    land_cover, _ = decode_grib(grib_bytes)

    # Formula lifted verbatim from ifs-scripts/clim-pproc/generate_land_mask.ksh
    # line 63. Keeping the ``* 1 + 0`` tail matches the ksh's typed output
    # (float scaling) rather than the boolean array numpy would emit for a
    # bare comparison.
    mask = evaluate_formula(
        "(land_cover > 0.5) * 1 + 0",
        {"land_cover": land_cover},
    )
    logger.info(
        "computed land mask: %d pixels above threshold out of %d",
        int((mask > 0).sum()),
        int(mask.size),
    )

    metadata: dict = {"packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    encoded = encode_grib(mask, grib_bytes, metadata=metadata)
    return {"land_mask": encoded}
