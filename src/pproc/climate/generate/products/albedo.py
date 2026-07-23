# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``albedo`` product: monthly six-component albedo on target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_albedo.ksh``. The ksh
loops over 6 source files (``month_aluvi``, ``month_aluvv``,
``month_aluvg``, ``month_alnii``, ``month_alniv``, ``month_alnig``); each
run of this product handles ONE of those files.

Two masking regimes selected by ``--regime``:

* ``land-only`` (aluvv, aluvg, alniv, alnig): monthly loop applies
  ``field * land_mask`` and re-encodes each month with the target
  paramId.
* ``three-mask`` (aluvi, alnii): monthly loop applies three separate
  formulae — glacier-free land, ocean, glacier — sums them, and encodes.

Equality translations (mir-compute ``=`` → pproc-formula ``==``):

* ``(field = 0)`` → ``(field == 0)`` — the "missing over land → 0.15" gate.
* ``(0.149 = field)`` → ``(0.149 == field)`` — the "exact-0.149 → 0.8" gate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import eccodes
import numpy as np
from conflator import CLIArg
from pydantic import ConfigDict, Field

from pproc.climate import mir_ops
from pproc.climate.generate.config import BaseGenerateConfig
from pproc.common.io import decode_grib, encode_grib
from pproc.formula import evaluate_formula

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "AlbedoConfig"]


FIELD_NAME = "albedo"
DESCRIPTION = "Monthly 6-component albedo on target grid (per-file, two regimes)."


logger = logging.getLogger(__name__)


class AlbedoConfig(BaseGenerateConfig):
    """Config for the albedo product."""

    model_config = ConfigDict(extra="forbid")

    albedo_in: Annotated[
        Path,
        CLIArg("--albedo-in", default=None),
        Field(description="One source albedo GRIB (e.g. ``month_aluvi``)."),
    ] = Path("./month_alb_component")

    land_mask_in: Annotated[
        Path,
        CLIArg("--land-mask-in", default=None),
        Field(description="Land mask on target grid."),
    ] = Path("./land_mask")

    glacier_mask_in: Annotated[
        Path,
        CLIArg("--glacier-mask-in", default=None),
        Field(
            description="Glacier mask on target grid (only used in three-mask regime)."
        ),
    ] = Path("./glacier_mask")

    glacier_free_land_mask_in: Annotated[
        Path,
        CLIArg("--glacier-free-land-mask-in", default=None),
        Field(
            description=(
                "Glacier-free land mask on target grid (only used in three-mask regime)."
            ),
        ),
    ] = Path("./glacier_free_land_mask")

    albedo_out: Annotated[
        Path,
        CLIArg("--albedo-out", default=None),
        Field(
            description="Output path for this component. Default ``./month_alb_out``."
        ),
    ] = Path("./month_alb_out")

    regime: Annotated[
        str,
        CLIArg("--regime", default=None),
        Field(
            description=(
                "``land-only`` for aluvv/aluvg/alniv/alnig; ``three-mask`` for "
                "aluvi/alnii (isotropic components)."
            ),
        ),
    ] = "land-only"

    paramId: Annotated[
        int,
        CLIArg("--paramId", type=int, default=None),
        Field(
            description=("Target GRIB paramId (ksh: 210186..210191 in order)."),
        ),
    ]

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = AlbedoConfig


def generate(config: AlbedoConfig) -> dict[str, bytes]:
    """Interpolate + apply the appropriate masking regime, month by month."""
    logger.info(
        "albedo: %s regime=%s paramId=%d → %s",
        config.albedo_in,
        config.regime,
        config.paramId,
        config.target_grid,
    )
    src = config.albedo_in.read_bytes()
    interpolated = mir_ops.interpolate(
        src, grid=config.target_grid, method="grid-box-average"
    )

    land_mask, _ = decode_grib(config.land_mask_in.read_bytes())
    if config.regime == "three-mask":
        glacier_mask, _ = decode_grib(config.glacier_mask_in.read_bytes())
        glacier_free_land_mask, _ = decode_grib(
            config.glacier_free_land_mask_in.read_bytes()
        )
    else:
        glacier_mask = None
        glacier_free_land_mask = None

    metadata: dict = {
        "paramId": config.paramId,
        "packingType": "grid_simple",
        # ksh used ``setBitsPerValue=16``; that is an eccodes accessor which
        # ultimately sets bitsPerValue on the message. encode_grib() has
        # dedicated handling for the bitsPerValue key (it's applied after
        # construct_message), so we pass it directly.
        "bitsPerValue": 16,
    }
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    out_chunks: list[bytes] = []
    reader = eccodes.MemoryReader(interpolated)
    n = 0
    for message in reader:
        wire = message.get_buffer()
        values = np.asarray(message.get_array("values"), dtype=np.float64)
        if config.regime == "land-only":
            masked = evaluate_formula(
                "field * land_mask",
                {"field": values, "land_mask": land_mask},
            )
        elif config.regime == "three-mask":
            # note: '=' equality rewritten to '==' for pproc-formula
            field_land = evaluate_formula(
                "field * land_mask + 0.15 * (field == 0) * land_mask",
                {"field": values, "land_mask": glacier_free_land_mask},
            )
            field_ocean = evaluate_formula(
                "0.06 * (1 - land_mask)",
                {"field": values, "land_mask": land_mask},
            )
            field_glacier = evaluate_formula(
                "field * glacier_mask * (field > 0.151) + "
                "0.15 * glacier_mask * (field < 0.149) + "
                "0.8 * glacier_mask * (0.149 <= field) * (field <= 0.151) + "
                "0.8 * (0.149 == field) * glacier_mask",
                {"field": values, "glacier_mask": glacier_mask},
            )
            masked = evaluate_formula(
                "field_land + field_ocean + field_glacier",
                {
                    "field_land": field_land,
                    "field_ocean": field_ocean,
                    "field_glacier": field_glacier,
                },
            )
        else:
            raise ValueError(f"unknown regime {config.regime!r}")

        out_chunks.append(encode_grib(masked, wire, metadata=metadata))
        n += 1
    logger.info("albedo: %d messages encoded", n)

    return {"albedo": b"".join(out_chunks)}
