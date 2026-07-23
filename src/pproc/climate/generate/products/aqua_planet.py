# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``aqua-planet`` product: monthly aqua-planet climatology on target grid.

Faithful port of ``ifs-scripts/clim-pproc/generate_aqua_planet.ksh``. Runs
nearest-neighbour interpolation on 7 lake fields; regrids the source
orography with nearest-lsm to the target grid; computes the orographic
correction term; and applies it to the 4 temperature-based lake fields
(``lakemlt``, ``lakeblt``, ``laketlt``, ``lakeict``) on a per-month basis.

The remaining 3 fields (``lakemld``, ``lakeshf``, ``lakeicd``) get the
metadata edit but no orographic correction.

The outer loop over the 7 lake fields is a loop over LOGICAL products (each
yields its own output), but they share the same orographic-correction term
computed once from the source/target orographies — so the entire 7-way loop
lives inside this product. All 7 outputs are returned.
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

__all__ = ["FIELD_NAME", "DESCRIPTION", "CONFIG", "generate", "AquaPlanetConfig"]


FIELD_NAME = "aqua-planet"
DESCRIPTION = "Monthly aqua-planet climatology (7 lake fields; orographic correction)."


logger = logging.getLogger(__name__)


_FIELDS = ["lakemlt", "lakemld", "lakeblt", "laketlt", "lakeshf", "lakeict", "lakeicd"]
_TEMP_FIELDS = {"lakemlt", "lakeblt", "laketlt", "lakeict"}


class AquaPlanetConfig(BaseGenerateConfig):
    """Config for the aqua-planet product."""

    model_config = ConfigDict(extra="forbid")

    source_dir: Annotated[
        Path,
        CLIArg("--source-dir", default=None),
        Field(
            description=(
                "Directory containing the 7 lake source files "
                "(``lakemlt``, ..., ``lakeicd``). ksh: ``${CLIMFIELDS_SOURCEDATA}``."
            ),
        ),
    ] = Path("./source")

    orog_source_in: Annotated[
        Path,
        CLIArg("--orog-source-in", default=None),
        Field(
            description="Source-grid orography+lsm bundle (ksh: /home/rdx/..../lsmoro)."
        ),
    ] = Path("./orog_source_lsmoro")

    orog_target_in: Annotated[
        Path,
        CLIArg("--orog-target-in", default=None),
        Field(description="Target-grid orography (ksh: ${XDATA_IFS}/lsmoro)."),
    ] = Path("./orog_target_lsmoro")

    land_mask_in: Annotated[
        Path,
        CLIArg("--land-mask-in", default=None),
        Field(description="Land mask on target grid."),
    ] = Path("./land_mask")

    lakemlt_out: Annotated[
        Path, CLIArg("--lakemlt-out", default=None), Field(description="lakemlt.")
    ] = Path("./lakemlt")
    lakemld_out: Annotated[
        Path, CLIArg("--lakemld-out", default=None), Field(description="lakemld.")
    ] = Path("./lakemld")
    lakeblt_out: Annotated[
        Path, CLIArg("--lakeblt-out", default=None), Field(description="lakeblt.")
    ] = Path("./lakeblt")
    laketlt_out: Annotated[
        Path, CLIArg("--laketlt-out", default=None), Field(description="laketlt.")
    ] = Path("./laketlt")
    lakeshf_out: Annotated[
        Path, CLIArg("--lakeshf-out", default=None), Field(description="lakeshf.")
    ] = Path("./lakeshf")
    lakeict_out: Annotated[
        Path, CLIArg("--lakeict-out", default=None), Field(description="lakeict.")
    ] = Path("./lakeict")
    lakeicd_out: Annotated[
        Path, CLIArg("--lakeicd-out", default=None), Field(description="lakeicd.")
    ] = Path("./lakeicd")

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(min_length=1, description="Target output grid."),
    ]


CONFIG = AquaPlanetConfig


def generate(config: AquaPlanetConfig) -> dict[str, bytes]:
    """Full aqua-planet pipeline. Returns 7 named outputs."""
    logger.info("aqua-planet: source=%s → %s", config.source_dir, config.target_grid)

    # ---- 2.1: extract source orography + lsm; regrid source orography with nearest-lsm
    src_lsmoro = config.orog_source_in.read_bytes()
    orog_input = _extract_by_paramid(src_lsmoro, 129)
    lsm_input = _extract_by_paramid(src_lsmoro, 172)
    if orog_input is None or lsm_input is None:
        raise ValueError(
            f"orog-source-in {config.orog_source_in} does not contain both "
            "paramId=129 (orography) and paramId=172 (lsm)"
        )

    orog_source = _interpolate_orog_source_nearest_lsm(orog_input, lsm_input, config)

    # ---- extract target orography
    target_lsmoro = config.orog_target_in.read_bytes()
    orog_target = _extract_by_paramid(target_lsmoro, 129)
    if orog_target is None:
        raise ValueError(
            f"orog-target-in {config.orog_target_in} does not contain paramId=129"
        )

    # ---- 2.2: orographic correction
    orog_source_vals, _ = decode_grib(orog_source)
    orog_target_vals, _ = decode_grib(orog_target)
    correction = evaluate_formula(
        "-0.0065 / 9.80665 * (orog_target - orog_source)",
        {"orog_source": orog_source_vals, "orog_target": orog_target_vals},
    )

    # ---- 1 + 2.3: interpolate each field, optionally correct
    outputs: dict[str, bytes] = {}
    for name in _FIELDS:
        in_path = config.source_dir / name
        src = in_path.read_bytes()
        interpolated = mir_ops.interpolate(
            src, grid=config.target_grid, method="nearest-neighbour"
        )
        if name in _TEMP_FIELDS:
            # Apply orographic correction on every monthly message.
            corrected_chunks: list[bytes] = []
            reader = eccodes.MemoryReader(interpolated)
            for message in reader:
                wire = message.get_buffer()
                values = np.asarray(message.get_array("values"), dtype=np.float64)
                corrected = evaluate_formula(
                    "field + orographic_correction",
                    {"field": values, "orographic_correction": correction},
                )
                corrected_chunks.append(_encode_final(corrected, wire, config))
            outputs[f"{name}"] = b"".join(corrected_chunks)
        else:
            # No formula — just re-pack with the aqua-planet metadata.
            chunks: list[bytes] = []
            reader = eccodes.MemoryReader(interpolated)
            for message in reader:
                wire = message.get_buffer()
                values = np.asarray(message.get_array("values"), dtype=np.float64)
                chunks.append(_encode_final(values, wire, config))
            outputs[f"{name}"] = b"".join(chunks)
        logger.info("aqua-planet: %s done", name)

    return outputs


def _extract_by_paramid(grib_bytes: bytes, paramid: int) -> "bytes | None":
    """Return the first message with the given paramId, or None."""
    reader = eccodes.MemoryReader(grib_bytes)
    for message in reader:
        try:
            if int(message.get("paramId")) == paramid:
                return message.get_buffer()
        except Exception:  # noqa: BLE001
            continue
    return None


def _interpolate_orog_source_nearest_lsm(
    orog_input_bytes: bytes, lsm_input_bytes: bytes, config: AquaPlanetConfig
) -> bytes:
    """Run nearest-lsm on the source orography via a tempfile round-trip.

    mir_ops.interpolate takes ``lsm_file_input`` / ``lsm_file_output`` as
    filesystem paths (mir doesn't expose an in-memory buffer API for the
    LSM inputs), so we stage the extracted lsm on disk in the CWD.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as f:
        f.write(lsm_input_bytes)
        lsm_source_path = f.name
    try:
        return mir_ops.interpolate(
            orog_input_bytes,
            grid=config.target_grid,
            method="nearest-lsm",
            lsm_selection="file",
            lsm_file_input=lsm_source_path,
            lsm_file_output=str(config.land_mask_in),
        )
    finally:
        Path(lsm_source_path).unlink(missing_ok=True)


def _encode_final(values, template: bytes, config: AquaPlanetConfig) -> bytes:
    """Encode with edition-2 + setBitsPerValue=16 + grid_simple metadata."""
    metadata: dict = {
        "localDefinitionNumber": 1,
        "edition": 2,
        "packingType": "grid_simple",
        "bitsPerValue": 16,  # ksh: setBitsPerValue=16
    }
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value
    return encode_grib(values, template, metadata=metadata)
