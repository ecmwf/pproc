# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``soil-type-hwsd`` product: soil type via van Genuchten triangle (HWSD).

Faithful port of ``ifs-scripts/clim-pproc/generate_soil_type_hwsd.ksh``.

Ten formulae + one land-mask application. Stages:

* Regrid ``soil_sand``, ``soil_clay``, ``soil_silt``, ``soil_oc`` to target
  grid with ``grid-box-average``.
* Compute ``soil_total = (sand + clay + silt + 1e-5) / 100``.
* Compute per-type fractions: ``<type> / soil_total``.
* Apply the van Genuchten classification via 6 formulae (soil types 1..5,
  then oceanic-carbon override to type 6).
* Sum + normalise unassigned pixels to type 1.
* Multiply by land mask.
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
from pproc.formula import evaluate_formula

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "SoilTypeHwsdConfig"]


FIELD_NAME = "soil-type-hwsd"
DESCRIPTION = "Soil type (HWSD, van Genuchten) on target grid."


logger = logging.getLogger(__name__)


class SoilTypeHwsdConfig(BaseGenerateConfig):
    """Config for the soil-type-hwsd product."""

    model_config = ConfigDict(extra="forbid")

    soil_sand_in: Annotated[
        Path, CLIArg("--soil-sand-in", default=None), Field(description="Source sand.")
    ] = Path("./soil_sand")
    soil_clay_in: Annotated[
        Path, CLIArg("--soil-clay-in", default=None), Field(description="Source clay.")
    ] = Path("./soil_clay")
    soil_silt_in: Annotated[
        Path, CLIArg("--soil-silt-in", default=None), Field(description="Source silt.")
    ] = Path("./soil_silt")
    soil_oc_in: Annotated[
        Path,
        CLIArg("--soil-oc-in", default=None),
        Field(description="Source organic carbon."),
    ] = Path("./soil_oc")

    land_mask_in: Annotated[
        Path,
        CLIArg("--land-mask-in", default=None),
        Field(description="Land mask on target grid."),
    ] = Path("./land_mask")

    slt_hwsd_out: Annotated[
        Path,
        CLIArg("--slt-hwsd-out", default=None),
        Field(description="Output path. Default ``./slt_hwsd``."),
    ] = Path("./slt_hwsd")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = SoilTypeHwsdConfig


def _gba(src_bytes: bytes, grid: str) -> bytes:
    """Grid-box-average interpolation helper."""
    return mir_ops.interpolate(src_bytes, grid=grid, method="grid-box-average")


def generate(config: SoilTypeHwsdConfig) -> dict[str, bytes]:
    """Compute the HWSD soil type on the target grid."""
    logger.info("soil-type-hwsd: 4 sources → %s", config.target_grid)

    # 1. Interpolate 4 sources to target grid.
    sand_bytes = _gba(config.soil_sand_in.read_bytes(), config.target_grid)
    clay_bytes = _gba(config.soil_clay_in.read_bytes(), config.target_grid)
    silt_bytes = _gba(config.soil_silt_in.read_bytes(), config.target_grid)
    oc_bytes = _gba(config.soil_oc_in.read_bytes(), config.target_grid)

    sand, _ = decode_grib(sand_bytes)
    clay, _ = decode_grib(clay_bytes)
    silt, _ = decode_grib(silt_bytes)
    oc, _ = decode_grib(oc_bytes)
    land_mask, _ = decode_grib(config.land_mask_in.read_bytes())

    # 2. Total soil cover (excl. oceanic carbon)
    # Original ksh variables were "sand;silt;clay" bound positionally to the
    # cat order "soil_sand_grid soil_clay_grid soil_silt_grid" — a
    # variable-name/positional mismatch preserved from the mir-compute
    # original. We reproduce the same expression here with the intended
    # arrays.
    soil_total = evaluate_formula(
        "(sand + clay + silt + 0.00001) / 100",
        {"sand": sand, "clay": clay, "silt": silt},
    )

    # 3. Fractions
    sand_frac = evaluate_formula(
        "soil_type / soil_total", {"soil_type": sand, "soil_total": soil_total}
    )
    clay_frac = evaluate_formula(
        "soil_type / soil_total", {"soil_type": clay, "soil_total": soil_total}
    )
    silt_frac = evaluate_formula(
        "soil_type / soil_total", {"soil_type": silt, "soil_total": soil_total}
    )

    # 4. Van Genuchten classification (5 formulae, one per soil-type index 1..5)
    slt_1 = evaluate_formula(
        "1 * (sand > 70) * (clay <= 18) + "
        "1 * (sand > 65) * (sand <= 70) * (clay <= (18 * (sand - 65) / 5) )",
        {"sand": sand_frac, "clay": clay_frac},
    )
    slt_2 = evaluate_formula(
        "2 * (sand > 30) * (sand <= 65) * (clay <= 34) + "
        "2 * (sand > 70) * (clay > 18) * (clay <= 34) + "
        "2 * (sand > 65) * (sand <= 70) * (clay > 18) * (clay <= 34) + "
        "2 * (sand > 65) * (sand <= 70) * (clay > (18 * (sand - 65) / 5 ) ) * (clay < 18) + "
        "2 * (sand > 15) * (sand <= 30) * (clay <= (34 * (sand - 15) / 15) )",
        {"sand": sand_frac, "clay": clay_frac},
    )
    slt_3 = evaluate_formula(
        "3 * (sand <= 15) * (silt > 70) + "
        "3 * (sand > 15) * (sand <= 30) * (silt >= 65) * (silt <= (100 - 35 * (sand - 15) / 15) )",
        {"sand": sand_frac, "silt": silt_frac},
    )
    slt_4 = evaluate_formula("4 * (clay <= 60) * (clay > 34)", {"clay": clay_frac})
    slt_5 = evaluate_formula("5 * (clay > 60)", {"clay": clay_frac})

    # 5. Sum into one field (5-input positional bundle f1..f5)
    soil_types = evaluate_formula(
        "f1 + f2 + f3 + f4 + f5",
        {"f1": slt_1, "f2": slt_2, "f3": slt_3, "f4": slt_4, "f5": slt_5},
    )

    # 6. Oceanic-carbon override → type 6
    soil_types_updated = evaluate_formula(
        "6 * (soil_types <= 2) * (oceanic_carbon > 10) + "
        "soil_types * (soil_types > 2) * (oceanic_carbon > 10) + "
        "soil_types * (oceanic_carbon <= 10)",
        {"soil_types": soil_types, "oceanic_carbon": oc},
    )

    # 7. Unassigned → type 1
    soil_types_all = evaluate_formula(
        "soil_types * (soil_types >= 1) + 1 * (soil_types < 1)",
        {"soil_types": soil_types_updated},
    )

    # 8. Land-mask application
    soil_types_complete = evaluate_formula(
        "soil_types * land_mask",
        {"soil_types": soil_types_all, "land_mask": land_mask},
    )

    metadata: dict = {"paramId": 43, "packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    encoded = encode_grib(soil_types_complete, sand_bytes, metadata=metadata)
    return {"slt_hwsd": encoded}
