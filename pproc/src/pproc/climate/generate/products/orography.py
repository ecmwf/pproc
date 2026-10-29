# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``orography`` product: conservatively interpolate source orography to target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_orography.ksh``. The ksh
runs ``pproc-interpol --grid=$MIR_GTYPE_SET --interpolation=grid-box-average``
then ``grib_set`` for shortName/localDefinition. Both are folded into this
product.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from conflator import CLIArg
from pydantic import ConfigDict, Field

import eccodes

from pproc.climate import mir_ops
from pproc.climate.generate.config import BaseGenerateConfig

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "OrographyConfig"]


FIELD_NAME = "orography"
DESCRIPTION = "Mean orography on target grid (grid-box-average interpolation)."


logger = logging.getLogger(__name__)


class OrographyConfig(BaseGenerateConfig):
    """Config for the orography product."""

    model_config = ConfigDict(extra="forbid")

    orog_in: Annotated[
        Path,
        CLIArg("--orog-in", default=None),
        Field(
            description=(
                "Path to source orography GRIB "
                "(ksh: ``${CLIMFIELDS_SOURCEDATA}/orog``)."
            ),
        ),
    ] = Path("./orog")

    orog_out: Annotated[
        Path,
        CLIArg("--orog-out", default=None),
        Field(
            description="Output path for mean orography on target grid. Default ``./orog``.",
        ),
    ] = Path("./orog")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(
            min_length=1, description="Target output grid (ksh: ``$MIR_GTYPE_SET``)."
        ),
    ]


CONFIG = OrographyConfig


def generate(config: OrographyConfig) -> dict[str, bytes]:
    """Interpolate source orography via grid-box-average and set metadata.

    The ksh original only does ``grib_set -s shortName=z,centre=98,...``
    (no ``packingType`` change) — so we clone the mir output message and
    apply the keys directly, preserving mir's native packing (grid_ieee
    on N-grids).
    """
    logger.info(
        "orography: reading %s and regridding to %s", config.orog_in, config.target_grid
    )
    src = config.orog_in.read_bytes()
    regridded = mir_ops.interpolate(
        src, grid=config.target_grid, method="grid-box-average"
    )

    # Clone the first (and only) message and apply metadata via eccodes so we
    # do not touch packing / values — mirrors ``grib_set`` semantics.
    reader = eccodes.MemoryReader(regridded)
    msg = next(iter(reader)).copy()
    keys = {
        "shortName": "z",
        "centre": 98,
        "subCentre": 0,
        "setLocalDefinition": 1,
        "localDefinitionNumber": 1,
    }
    if config.bits_per_value is not None:
        keys["bitsPerValue"] = config.bits_per_value
    msg.set(keys, check_values=True)
    return {"orog": msg.get_buffer()}
