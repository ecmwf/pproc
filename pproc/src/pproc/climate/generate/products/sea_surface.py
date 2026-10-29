# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``sea-surface`` product: SST + sea-ice cover climatology on target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_sea_surface.ksh``. The
input ``sstgrib.clim.new`` carries daily messages of two shortNames
(``ci`` and ``stl1``) for a leap year (~732 messages). The ksh:

1. Runs ONE nearest-lsm interpolation on the whole file.
2. For each day, applies ``field * ocean_mask + (1 - ocean_mask) * 9999`` to
   each of the two per-day fields.
3. Grib-sets ``localDefinitionNumber=1, yearOfCentury=255,
   packingType=grid_simple`` on the whole thing.
4. Splits back into 12 monthly files by month.

Steps 1-4 are all one algorithm; they run inside this product. The 12
monthly outputs are returned under logical names ``sst_01`` … ``sst_12``.
The CLI layer's :func:`pproc.climate.generate.io.write_outputs` maps
those names onto the output paths using a single ``--sst-out-template``
flag whose value contains a ``{month:02d}`` (or ``{month}``) placeholder,
e.g. ``./sstgrib.clim.{month:02d}``. The wrapper carries the operational
filename convention as a single template string instead of twelve
explicit paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import eccodes
import numpy as np
from conflator import CLIArg
from pydantic import ConfigDict, Field, field_validator

from pproc.climate import mir_ops
from pproc.climate.generate.config import BaseGenerateConfig
from pproc.common.io import decode_grib, encode_grib
from pproc.formula import evaluate_formula

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "SeaSurfaceConfig"]


FIELD_NAME = "sea-surface"
DESCRIPTION = "Monthly SST + sea-ice on target grid (nearest-lsm + ocean mask)."


logger = logging.getLogger(__name__)


class SeaSurfaceConfig(BaseGenerateConfig):
    """Config for the sea-surface product.

    Twelve ``sst_MM_out`` fields (one per month) are declared explicitly so the
    CLI layer's ``write_outputs`` can locate each of the 12 monthly outputs.
    """

    model_config = ConfigDict(extra="forbid")

    sst_in: Annotated[
        Path,
        CLIArg("--sst-in", default=None),
        Field(description="Source SST + sea-ice climatology GRIB (many messages)."),
    ] = Path("./sstgrib.clim.new")

    source_lsm_in: Annotated[
        Path,
        CLIArg("--source-lsm-in", default=None),
        Field(description="Source land-sea mask GRIB."),
    ] = Path("./source_lsm")

    land_mask_in: Annotated[
        Path,
        CLIArg("--land-mask-in", default=None),
        Field(description="Land mask on target grid (for mir --lsm-file-output)."),
    ] = Path("./land_mask")

    ocean_mask_in: Annotated[
        Path,
        CLIArg("--ocean-mask-in", default=None),
        Field(description="Ocean mask on target grid (for the masking formula)."),
    ] = Path("./ocean_mask")

    sst_out_template: Annotated[
        str,
        CLIArg("--sst-out-template", default=None),
        Field(
            description=(
                "Filename template for the twelve monthly outputs. Must "
                "contain a ``{month}`` placeholder (typically "
                "``{month:02d}`` for zero-padded month numbers). "
                "Example: ``./sstgrib.clim.{month:02d}`` produces "
                "``sstgrib.clim.01`` through ``sstgrib.clim.12``. The "
                "wrapper owns the operational filename convention as a "
                "single template string; the tool substitutes the month "
                "number for each of the twelve outputs."
            ),
        ),
    ] = "./sstgrib.clim.{month:02d}"

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]

    @field_validator("sst_out_template")
    @classmethod
    def _template_must_contain_month_placeholder(cls, v: str) -> str:
        """Reject templates that lack the ``{month}`` placeholder.

        Without the placeholder, every monthly write would land at the
        same path (silent overwrite). Catching this at config-time
        surfaces the mistake immediately rather than after the run.
        """
        if "{month" not in v:
            raise ValueError(
                f"--sst-out-template must contain a '{{month}}' "
                f"placeholder (typically '{{month:02d}}'); got {v!r}"
            )
        return v


CONFIG = SeaSurfaceConfig


def generate(config: SeaSurfaceConfig) -> dict[str, bytes]:
    """Full sea-surface pipeline; returns up to 12 monthly outputs."""
    logger.info(
        "sea-surface: %s → %s (nearest-lsm, then daily mask)",
        config.sst_in,
        config.target_grid,
    )
    src = config.sst_in.read_bytes()

    interpolated = mir_ops.interpolate(
        src,
        grid=config.target_grid,
        method="nearest-lsm",
        lsm_selection="file",
        lsm_file_input=str(config.source_lsm_in),
        lsm_file_output=str(config.land_mask_in),
    )

    ocean_mask, _ = decode_grib(config.ocean_mask_in.read_bytes())

    per_month: dict[str, list[bytes]] = {f"{m:02d}": [] for m in range(1, 13)}
    metadata_common: dict = {
        "localDefinitionNumber": 1,
        "yearOfCentury": 255,
        "packingType": "grid_simple",
    }
    if config.bits_per_value is not None:
        metadata_common["bitsPerValue"] = config.bits_per_value

    reader = eccodes.MemoryReader(interpolated)
    n = 0
    for message in reader:
        wire = message.get_buffer()
        values = np.asarray(message.get_array("values"), dtype=np.float64)
        masked = evaluate_formula(
            "field * (ocean_mask) + (1 - ocean_mask) * 9999",
            {"field": values, "ocean_mask": ocean_mask},
        )
        date = int(message.get("dataDate"))
        mm = f"{(date // 100) % 100:02d}"
        per_month[mm].append(encode_grib(masked, wire, metadata=metadata_common))
        n += 1
    logger.info("sea-surface: masked %d daily messages", n)

    outputs: dict[str, bytes] = {}
    for mm, chunks in per_month.items():
        if chunks:
            outputs[f"sst_{mm}"] = b"".join(chunks)
    return outputs
