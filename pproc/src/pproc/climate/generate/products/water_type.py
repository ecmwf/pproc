# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``water-type`` product: categorical water-type mask.

Faithful port of ``ifs-scripts/clim-pproc/generate_water_type.ksh``.

Starts from ``ocean_mask`` (label 1) and iterates over N waterbody covers
(``waterbody_c2``, optionally ``waterbody_c3``, ...). Each waterbody
contributes label ``i+1`` for pixels where its cover >= 0.5:

    types = ocean_mask
    for i, wb in enumerate(waterbodies, start=1):
        mask = (wb_cover >= 0.5) * 1 + 0
        types = types + (i + 1) * mask

Final metadata: ``paramId=172, packingType=grid_simple``.

No interpolation; all inputs must be on the target grid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, List

from conflator import CLIArg
from pydantic import ConfigDict, Field

from pproc.climate.generate.config import BaseGenerateConfig
from pproc.common.io import decode_grib, encode_grib
from pproc.formula import evaluate_formula

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "WaterTypeConfig"]


FIELD_NAME = "water-type"
DESCRIPTION = "Categorical water-type field (ocean + N waterbodies) on target grid."


logger = logging.getLogger(__name__)


class WaterTypeConfig(BaseGenerateConfig):
    """Config for the water-type product."""

    model_config = ConfigDict(extra="forbid")

    ocean_mask_in: Annotated[
        Path,
        CLIArg("--ocean-mask-in", default=None),
        Field(description="Ocean mask on target grid (label 1)."),
    ] = Path("./ocean_mask")

    waterbody_cover_in: Annotated[
        List[Path],
        CLIArg("--waterbody-cover-in", nargs="+", default=None),
        Field(
            description=(
                "One or more waterbody-cover GRIBs on target grid, in label "
                "order (label 2 = first, label 3 = second, ...). ksh loops over "
                "``waterbody_c2`` and (optionally) ``waterbody_c3``, ..."
            ),
        ),
    ] = [Path("./waterbody_c2")]

    water_type_out: Annotated[
        Path,
        CLIArg("--water-type-out", default=None),
        Field(description="Output path. Default ``./water_type``."),
    ] = Path("./water_type")


CONFIG = WaterTypeConfig


def generate(config: WaterTypeConfig) -> dict[str, bytes]:
    """Compose the categorical water-type field."""
    ocean_bytes = config.ocean_mask_in.read_bytes()
    ocean, _ = decode_grib(ocean_bytes)
    types = ocean.copy()  # label 1 = ocean

    for idx, wb_path in enumerate(config.waterbody_cover_in, start=2):
        wb_cover, _ = decode_grib(Path(wb_path).read_bytes())
        # Threshold (>= 0.5) — same formula as lake-mask
        new_type = evaluate_formula(
            "(waterbody_cover >= 0.5) * 1 + 0",
            {"waterbody_cover": wb_cover},
        )
        # In-place accumulation: types = types + label * new_type
        types = evaluate_formula(
            f"types + {idx} * new_type",
            {"types": types, "new_type": new_type},
        )
        logger.info("water-type: applied label=%d from %s", idx, wb_path)

    metadata: dict = {"paramId": 172, "packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    encoded = encode_grib(types, ocean_bytes, metadata=metadata)
    return {"water_type": encoded}
