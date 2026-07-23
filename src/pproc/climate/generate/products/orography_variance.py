# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``orography-variance`` product: total variance of orography on target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_orography_variance.ksh``.

Total variance = mean(variance) + variance(mean):

1. Regrid 30" orography variance → mean(variance) on target grid.
2. Regrid 30" orography → mean(x) on target grid.
3. Compute x² pointwise (source resolution).
4. Regrid x² → mean(x²) on target grid.
5. variance(mean) = ``abs(mean_square - mean * mean)``.
6. total = ``mean_var + var_of_mean``.
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

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "OrographyVarianceConfig"]


FIELD_NAME = "orography-variance"
DESCRIPTION = "Total orographic variance on target grid (law of total variance)."


logger = logging.getLogger(__name__)


class OrographyVarianceConfig(BaseGenerateConfig):
    """Config for the orography-variance product."""

    model_config = ConfigDict(extra="forbid")

    orog_in: Annotated[
        Path,
        CLIArg("--orog-in", default=None),
        Field(description='30" orography (sub-km mean).'),
    ] = Path("./orog")

    orog_variance_in: Annotated[
        Path,
        CLIArg("--orog-variance-in", default=None),
        Field(description='30" orography sub-km variance.'),
    ] = Path("./orog_variance")

    orog_variance_out: Annotated[
        Path,
        CLIArg("--orog-variance-out", default=None),
        Field(description="Output path. Default ``./orog_variance``."),
    ] = Path("./orog_variance")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = OrographyVarianceConfig


def generate(config: OrographyVarianceConfig) -> dict[str, bytes]:
    """Compute total variance via the law of total variance."""
    logger.info("orography-variance: → %s", config.target_grid)
    orog_var_src = config.orog_variance_in.read_bytes()
    orog_src = config.orog_in.read_bytes()

    # 1. mean_var on target grid
    mean_subkm_var_bytes = mir_ops.interpolate(
        orog_var_src, grid=config.target_grid, method="grid-box-average"
    )
    mean_subkm_var, _ = decode_grib(mean_subkm_var_bytes)

    # 2. mean(x) on target
    orog_mean_bytes = mir_ops.interpolate(
        orog_src, grid=config.target_grid, method="grid-box-average"
    )
    orog_mean, _ = decode_grib(orog_mean_bytes)

    # 3. x^2 pointwise on source grid
    orog_vals, _ = decode_grib(orog_src)
    orog_sq = evaluate_formula("field * field", {"field": orog_vals})
    orog_sq_bytes = encode_grib(orog_sq, orog_src)

    # 4. mean(x^2) on target grid
    mean_orog_sq_bytes = mir_ops.interpolate(
        orog_sq_bytes, grid=config.target_grid, method="grid-box-average"
    )
    mean_orog_sq, _ = decode_grib(mean_orog_sq_bytes)

    # 5. variance(mean) = abs(mean_square - mean^2)
    orog_km_var = evaluate_formula(
        "abs( mean_square - (mean * mean) )",
        {"mean_square": mean_orog_sq, "mean": orog_mean},
    )

    # 6. total = mean_var + var_of_mean
    total = evaluate_formula(
        "mean_var + var_of_mean",
        {"mean_var": mean_subkm_var, "var_of_mean": orog_km_var},
    )
    logger.info(
        "orography-variance: total var min=%.3f max=%.3f mean=%.3f",
        float(total.min()),
        float(total.max()),
        float(total.mean()),
    )

    metadata: dict = {"paramId": 200, "packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    encoded = encode_grib(total, mean_subkm_var_bytes, metadata=metadata)
    return {"orog_variance": encoded}
