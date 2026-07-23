# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``albedo-four-stream`` product: 4 direct/diffuse components + OSM albedo.

Faithful port of ``ifs-scripts/clim-pproc/generate_albedo_four_stream.ksh``.

Two invocation modes:

* ``per-component`` — regrid one of ``month_aluvp``, ``month_aluvd``,
  ``month_alnip``, ``month_alnid``; on each monthly message apply the
  land-mask + ocean-mask formulae ((field==0) → 0.15 fill on land;
  1-land_mask → 0.06 ocean; sum) and set paramId. One output.
* ``osm`` — read ``month_alnid`` and ``month_aluvd`` already processed
  on the target grid; on each month, compute
  ``0.45976 * aluvd + 0.54024 * alnid`` and encode with ``paramId=174``.

The wrapper drives the mode with ``--mode``; the outer loop over the four
component files (LOGICAL products) stays in the wrapper.

Equality translation: ``(field = 0)`` → ``(field == 0)``.
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

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "AlbedoFourStreamConfig"]


FIELD_NAME = "albedo-four-stream"
DESCRIPTION = "4-component albedo + monthly OSM albedo on target grid."


logger = logging.getLogger(__name__)


class AlbedoFourStreamConfig(BaseGenerateConfig):
    """Config for the albedo-four-stream product."""

    model_config = ConfigDict(extra="forbid")

    mode: Annotated[
        str,
        CLIArg("--mode", default=None),
        Field(
            description=(
                "``per-component`` for a single 4-stream component; ``osm`` for "
                "the derived offline-surface-model albedo."
            ),
        ),
    ] = "per-component"

    albedo_in: Annotated[
        Path,
        CLIArg("--albedo-in", default=None),
        Field(
            description=(
                "For ``per-component``: source GRIB (e.g. ``month_aluvp``). "
                "Ignored in ``osm`` mode."
            ),
        ),
    ] = Path("./month_alb_component")

    alnid_in: Annotated[
        Path,
        CLIArg("--alnid-in", default=None),
        Field(description="For ``osm``: target-grid ``month_alnid``."),
    ] = Path("./month_alnid")

    aluvd_in: Annotated[
        Path,
        CLIArg("--aluvd-in", default=None),
        Field(description="For ``osm``: target-grid ``month_aluvd``."),
    ] = Path("./month_aluvd")

    land_mask_in: Annotated[
        Path,
        CLIArg("--land-mask-in", default=None),
        Field(description="Land mask on target grid (``per-component`` only)."),
    ] = Path("./land_mask")

    albedo_out: Annotated[
        Path,
        CLIArg("--albedo-out", default=None),
        Field(description="Output GRIB path. Default ``./month_alb_out``."),
    ] = Path("./month_alb_out")

    paramId: Annotated[
        int,
        CLIArg("--paramId", type=int, default=None),
        Field(
            description=(
                "Target paramId. For ``per-component`` the ksh iterates 15..18 "
                "(aluvp, aluvd, alnip, alnid); for ``osm`` it is 174."
            ),
        ),
    ] = 174

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(description="Target output grid (``per-component`` mode only)."),
    ] = ""


CONFIG = AlbedoFourStreamConfig


def generate(config: AlbedoFourStreamConfig) -> dict[str, bytes]:
    """Dispatch to per-component or osm mode."""
    if config.mode == "per-component":
        return _per_component(config)
    if config.mode == "osm":
        return _osm(config)
    raise ValueError(f"unknown --mode {config.mode!r}")


def _per_component(config: AlbedoFourStreamConfig) -> dict[str, bytes]:
    if not config.target_grid:
        raise ValueError("--target-grid is required in per-component mode")
    logger.info(
        "albedo-four-stream per-component: %s paramId=%d → %s",
        config.albedo_in,
        config.paramId,
        config.target_grid,
    )
    src = config.albedo_in.read_bytes()
    interpolated = mir_ops.interpolate(
        src, grid=config.target_grid, method="grid-box-average"
    )
    land_mask, _ = decode_grib(config.land_mask_in.read_bytes())

    metadata: dict = {
        "paramId": config.paramId,
        "packingType": "grid_simple",
        "bitsPerValue": 16,  # ksh: setBitsPerValue=16
    }
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value

    out_chunks: list[bytes] = []
    reader = eccodes.MemoryReader(interpolated)
    n = 0
    for message in reader:
        wire = message.get_buffer()
        values = np.asarray(message.get_array("values"), dtype=np.float64)
        # note: '=' equality rewritten to '==' for pproc-formula
        field_land = evaluate_formula(
            "field * land_mask + 0.15 * (field == 0) * land_mask",
            {"field": values, "land_mask": land_mask},
        )
        field_ocean = evaluate_formula(
            "0.06 * (1 - land_mask)",
            {"field": values, "land_mask": land_mask},
        )
        masked = evaluate_formula(
            "field_land + field_ocean",
            {"field_land": field_land, "field_ocean": field_ocean},
        )
        out_chunks.append(encode_grib(masked, wire, metadata=metadata))
        n += 1
    logger.info("albedo-four-stream per-component: %d messages", n)
    return {"albedo": b"".join(out_chunks)}


def _osm(config: AlbedoFourStreamConfig) -> dict[str, bytes]:
    """Compute month_alb_osm = 0.45976 * aluvd + 0.54024 * alnid (per month).

    The ksh's grib_set for OSM only touches paramId (no packing). To preserve
    the target-grid input's packing (grid_simple, bpv=16) we clone the
    alnid template and set only paramId + values — matching ``grib_set``.
    """
    logger.info("albedo-four-stream osm: %s + %s", config.alnid_in, config.aluvd_in)
    alnid_bytes = config.alnid_in.read_bytes()
    aluvd_bytes = config.aluvd_in.read_bytes()

    aluvd_by_date: dict[int, bytes] = {}
    for m in eccodes.MemoryReader(aluvd_bytes):
        aluvd_by_date[int(m.get("dataDate"))] = m.get_buffer()

    out_chunks: list[bytes] = []
    n = 0
    for msg in eccodes.MemoryReader(alnid_bytes):
        date = int(msg.get("dataDate"))
        if date not in aluvd_by_date:
            raise ValueError(
                f"osm: aluvd has no message for dataDate={date} (present in alnid)"
            )
        alnid_vals = np.asarray(msg.get_array("values"), dtype=np.float64)
        aluvd_vals, _ = decode_grib(aluvd_by_date[date])
        osm = evaluate_formula(
            "0.45976 * aluvd + 0.54024 * alnid",
            {"aluvd": aluvd_vals, "alnid": alnid_vals},
        )
        # Clone + set only paramId + values, preserving packing.
        out_msg = msg.copy()
        keys: dict = {"paramId": config.paramId}
        if config.bits_per_value is not None:
            keys["bitsPerValue"] = config.bits_per_value
        out_msg.set(keys, check_values=True)
        out_msg.set_array("values", np.asarray(osm, dtype=np.float64).copy())
        out_chunks.append(out_msg.get_buffer())
        n += 1
    logger.info("albedo-four-stream osm: %d months", n)
    return {"albedo": b"".join(out_chunks)}
