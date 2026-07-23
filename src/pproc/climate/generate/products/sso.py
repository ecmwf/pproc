# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``sso`` product: sub-grid scale orography pipeline.

A faithful Python port of ``ifs-scripts/clim-pproc/generate_subgrid_orography_sso.ksh``
(the pipeline logic was previously in ``pproc.climate.sso.pipeline`` and has
been folded into this module unchanged — see ``.weave/learnings/sso-migration.md``
for the ten-stage decomposition and the eleven mir-compute formulae).

Result key → shortName mapping:

================  =========  ================
Result key        shortName  Description
================  =========  ================
``stdgwd``        ``sdor``   Std. dev. of subgrid orography
``slogwd``        ``slor``   Std. dev. of slope
``anggwd``        ``anor``   Orientation
``isogwd``        ``isor``   Anisotropy
================  =========  ================

The pipeline supports two opt-in modes on :class:`SSOGenerateConfig`:

* ``grib_roundtrip=True`` — encode/decode every numpy result through
  GRIB before it leaves a stage. Reproduces the per-step GRIB
  quantisation that the original ksh applied via the ``cat`` +
  ``mir-compute`` pattern.
* ``dump_intermediates=True`` — write the 16 named intermediates to
  ``config.intermediates_dir`` using their canonical ksh filenames.
  Debug output only: filenames are hard-coded (no user-supplied path
  component), but the directory is configurable via
  ``--intermediates-dir``.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Iterator, Optional, Tuple

import numpy as np
from conflator import CLIArg
from pydantic import Field, model_validator

from pproc.climate import mir_ops
from pproc.climate.generate.config import BaseGenerateConfig
from pproc.climate.generate.effective_resolution import (
    compute_effective_resolution,
    infer_grid_params,
)
from pproc.common.io import decode_grib, encode_grib

__all__ = [
    "FIELD_NAME",
    "DESCRIPTION",
    "CONFIG",
    "SSOGenerateConfig",
    "generate",
    "compute_sso",
]


FIELD_NAME = "sso"
DESCRIPTION = (
    "Sub-grid scale orography: stdgwd/slogwd/anggwd/isogwd via the 10-stage pipeline."
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SSOGenerateConfig(BaseGenerateConfig):
    """Configuration for the SSO product.

    Subclasses :class:`~pproc.climate.generate.config.BaseGenerateConfig`
    (inheriting ``target_grid``, ``verbose``, ``grib_roundtrip``,
    ``bits_per_value``) and adds the SSO-specific input/output paths and
    grid knobs.

    Field-to-env-var mapping (see ``.weave/learnings/sso-migration.md``):

    ================================  ==========================================
    Field                             ksh env var / source
    ================================  ==========================================
    ``orography``                     ``$inFile``
    ``alt_orography``                 ``$inFile_alt``
    ``land_mask``                     ``$maskFile``
    ``target_grid``  (from base)      ``$MIR_GTYPE_SET``
    ``model_grid_type``               ``$GTYPE_SET``
    ``model_resolution``              ``$ORES``
    ``orography_grid``                hardcoded ``N2000`` in the ksh script
    ``effective_resolution``          ``$MIR_ERES_SET`` (derived)
    ``stdgwd_out`` / ``slogwd_out`` / per-output-path replacement for the ksh
    ``anggwd_out`` / ``isogwd_out``   ``$OUTPUT_DIR``
    ``bits_per_value`` (from base)    (no env-var; ksh passes 32)
    ================================  ==========================================
    """

    # extra="forbid" is inherited from BaseGenerateConfig? — no, ConfigModel
    # itself does not set extra. Pin it here so unknown fields on YAML
    # configs are caught loudly (matches the legacy SSOConfig behaviour).
    model_config = {
        "extra": "forbid",
        "revalidate_instances": "always",
        "validate_assignment": True,
        "validate_default": True,
    }

    # --- Inputs --------------------------------------------------------

    orography: Annotated[
        Path,
        CLIArg("--orography", default=None),
        Field(
            description="Source orography GRIB file (ksh: $inFile).",
        ),
    ]

    alt_orography: Annotated[
        Optional[Path],
        CLIArg("--alt-orography", default=None),
        Field(
            description=(
                "Alternative orography input. Used as a fallback when "
                "``orography`` does not exist on disk: the alternative is "
                "regridded to ``orography_grid`` and the result is cached "
                "at the ``orography`` path. Matches the ksh's ``$inFile_alt``."
            ),
        ),
    ] = None

    land_mask: Annotated[
        Path,
        CLIArg("--land-mask", default=None),
        Field(
            description="Land mask GRIB on target grid (ksh: $maskFile).",
        ),
    ]

    # --- Grid configuration --------------------------------------------
    # ``target_grid`` is redeclared here to make it *required* (the base
    # has it Optional). Conflator picks up the redeclared field cleanly.

    target_grid: Annotated[
        str,
        CLIArg("--target-grid", default=None),
        Field(
            min_length=1,
            description="Target output grid (ksh: $MIR_GTYPE_SET, e.g. 'N256').",
        ),
    ]

    model_grid_type: Annotated[
        str,
        CLIArg("--model-grid-type", default=None),
        Field(
            description=(
                "Model grid family code (ksh: $GTYPE_SET, e.g. 'O' or 'N'). "
                "Auto-inferred from ``target_grid`` when both this and "
                "``model_resolution`` are left at their defaults."
            ),
        ),
    ] = ""

    model_resolution: Annotated[
        int,
        CLIArg("--model-resolution", type=int, default=None),
        Field(
            ge=0,
            description=(
                "Model nominal resolution (ksh: $ORES, e.g. 80). "
                "Auto-inferred from ``target_grid`` when both this and "
                "``model_grid_type`` are left at their defaults."
            ),
        ),
    ] = 0

    orography_grid: Annotated[
        str,
        CLIArg("--orography-grid", default=None),
        Field(
            min_length=1,
            description=(
                "High-resolution working grid where SSO statistics are "
                "computed. Operationally ``N2000`` (≈ 5 km, hardcoded in "
                "the legacy ksh); tests use ``N256`` to keep fixtures small."
            ),
        ),
    ]

    effective_resolution: Annotated[
        str,
        CLIArg("--effective-resolution", default=None),
        Field(
            description=(
                "Effective-resolution grid (ksh: $MIR_ERES_SET). Always "
                "computed by ``resolve()`` from the model grid."
            ),
        ),
    ] = ""

    # --- Outputs (one Path per logical output) -------------------------

    stdgwd_out: Annotated[
        Path,
        CLIArg("--stdgwd-out", default=None),
        Field(
            description=(
                "Path for stdgwd output GRIB (shortName=sdor). Default ``./stdgwd``."
            ),
        ),
    ] = Path("./stdgwd")

    slogwd_out: Annotated[
        Path,
        CLIArg("--slogwd-out", default=None),
        Field(
            description=(
                "Path for slogwd output GRIB (shortName=slor). Default ``./slogwd``."
            ),
        ),
    ] = Path("./slogwd")

    anggwd_out: Annotated[
        Path,
        CLIArg("--anggwd-out", default=None),
        Field(
            description=(
                "Path for anggwd output GRIB (shortName=anor). Default ``./anggwd``."
            ),
        ),
    ] = Path("./anggwd")

    isogwd_out: Annotated[
        Path,
        CLIArg("--isogwd-out", default=None),
        Field(
            description=(
                "Path for isogwd output GRIB (shortName=isor). Default ``./isogwd``."
            ),
        ),
    ] = Path("./isogwd")

    # --- Debug knobs ----------------------------------------------------

    dump_intermediates: Annotated[
        bool,
        CLIArg("--dump-intermediates", action="store_true", default=None),
        Field(
            description=(
                "Write the 16 named intermediate GRIB files to "
                "``intermediates_dir`` (debug only)."
            ),
        ),
    ] = False

    intermediates_dir: Annotated[
        Path,
        CLIArg("--intermediates-dir", default=None),
        Field(
            description=(
                "Directory for the 16 named intermediates when "
                "``--dump-intermediates`` is set. Filenames are "
                "algorithm-intrinsic (hard-coded); only the directory "
                "is user-controlled. Default ``.``."
            ),
        ),
    ] = Path(".")

    # ------------------------------------------------------------------

    def resolve(self) -> "SSOGenerateConfig":
        """Return a copy with all auto-inferred fields populated.

        Idempotent. Inference precedence: if the caller supplied BOTH
        ``model_grid_type`` and ``model_resolution`` explicitly, take
        them as given; otherwise infer both from ``target_grid``.
        ``effective_resolution`` is always recomputed from the resolved
        model grid.
        """
        explicit_model = bool(self.model_grid_type) and self.model_resolution > 0

        if explicit_model:
            grid_type = self.model_grid_type
            resolution = self.model_resolution
        else:
            grid_type, resolution = infer_grid_params(self.target_grid)

        eff_res = compute_effective_resolution(grid_type, resolution)

        return self.model_copy(
            update={
                "model_grid_type": grid_type,
                "model_resolution": resolution,
                "effective_resolution": eff_res,
            }
        )

    @model_validator(mode="after")
    def _paths_are_paths(self) -> "SSOGenerateConfig":
        """Ensure Path-typed fields survive dict inputs as ``pathlib.Path``.

        Pydantic's Path coercion covers most cases, but round-tripping
        via ``model_dump(mode='json')`` produces strings; this validator
        keeps ``SSOGenerateConfig(**cfg.model_dump()).orography ==
        Path(...)`` invariant across YAML/JSON boundaries.
        """
        # Nothing to do — Path fields are already coerced by pydantic.
        # Kept as a hook for future path-shape checks.
        return self


CONFIG = SSOGenerateConfig


# ---------------------------------------------------------------------------
# Pipeline helpers (moved verbatim from pproc.climate.sso.pipeline; the only
# adjustments below are that ``output_dir`` → ``intermediates_dir`` for the
# intermediate dumps, and metadata/write of the four outputs now happens
# by returning bytes rather than writing to disk).
# ---------------------------------------------------------------------------


@contextmanager
def _stage(label: str) -> Iterator[None]:
    """Emit ``stage <label>`` on entry and ``stage <label> complete elapsed=...``
    on exit. Uses :func:`time.monotonic` so the delta is unaffected by wall-clock
    jitter or NTP adjustments mid-run."""
    logger.info("stage %s", label)
    t0 = time.monotonic()
    try:
        yield
    finally:
        stage_id = label.split(" ", 1)[0]
        logger.info("stage %s complete elapsed=%.3f", stage_id, time.monotonic() - t0)


def _log_array(stage_name: str, arr: np.ndarray) -> None:
    """DEBUG-only sketch of a numpy intermediate."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "%s array shape=%s dtype=%s bytes=%d",
            stage_name,
            tuple(arr.shape),
            arr.dtype,
            int(arr.nbytes),
        )


def _decode(grib_bytes: bytes) -> Tuple[np.ndarray, bytes]:
    """Decode the first message; return ``(values, template_bytes)``.

    Returning the raw ``grib_bytes`` as template is fine — ``encode_grib``
    only reads the first message of a multi-message buffer.
    """
    values, _ = decode_grib(grib_bytes)
    return values, grib_bytes


def _maybe_roundtrip(
    values: np.ndarray,
    template: bytes,
    config: SSOGenerateConfig,
    *,
    stage: str = "",
) -> np.ndarray:
    """Apply per-step GRIB quantisation when ``grib_roundtrip`` is on."""
    if not config.grib_roundtrip:
        return values
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "roundtrip stage=%s array shape=%s bytes=%d",
            stage or "?",
            tuple(values.shape),
            int(values.nbytes),
        )
    encoded = encode_grib(values, template)
    decoded, _ = decode_grib(encoded)
    return decoded


def _write_intermediate(name: str, payload: bytes, config: SSOGenerateConfig) -> None:
    """Persist a named intermediate when ``dump_intermediates`` is on.

    ``name`` is hard-coded at every call site (matches the ksh's canonical
    filenames) — so the joined path cannot escape ``intermediates_dir``
    via traversal. The directory is created on demand.
    """
    if not config.dump_intermediates:
        return
    config.intermediates_dir.mkdir(parents=True, exist_ok=True)
    target = config.intermediates_dir / name
    target.write_bytes(payload)
    logger.info("wrote intermediate %s → %s (%d bytes)", name, target, len(payload))


def _encode_on_template(values: np.ndarray, template: bytes) -> bytes:
    return encode_grib(values, template)


def _output_metadata(short_name: str, config: SSOGenerateConfig) -> dict:
    """Build the output metadata dict, optionally pinning bitsPerValue."""
    metadata: dict = {"shortName": short_name, "packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value
    return metadata


# ---------------------------------------------------------------------------
# Stage helpers — one per row of the ten-stage table
# ---------------------------------------------------------------------------


def _stage_source_to_orography_grid(config: SSOGenerateConfig) -> bytes:
    """Stage 1 — source orography lifted onto ``config.orography_grid``.

    Three cases:

    1. Fast path: ``config.orography`` exists and matches
       ``config.orography_grid`` — bytes pass through.
    2. Grid mismatch: file exists but on a different grid → ValueError
       (operators should move to ``--alt-orography`` for regrid+cache).
    3. Fallback: ``config.orography`` missing → regrid ``alt_orography``
       to ``orography_grid`` and cache-writeback to the ``orography``
       path (matches the ksh's ``inFile`` / ``inFile_alt`` pattern).
    """
    if config.orography.is_file():
        grib_bytes = config.orography.read_bytes()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "stage 1 reading %s (%d bytes)",
                config.orography,
                len(grib_bytes),
            )
        _, metadata = decode_grib(grib_bytes)
        input_grid = metadata.get("gridName")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "stage 1 decoded gridName=%s; comparing against "
                "config.orography_grid=%s",
                input_grid,
                config.orography_grid,
            )
        if input_grid == config.orography_grid:
            logger.info("stage 1 fast path (input on %s)", config.orography_grid)
            return grib_bytes
        raise ValueError(
            f"orography file '{config.orography}' is on grid "
            f"'{input_grid}' but --orography-grid is "
            f"'{config.orography_grid}'; supply an orography on "
            f"'{config.orography_grid}', or move this file to "
            f"--alt-orography to have it regridded."
        )

    if config.alt_orography is not None:
        if not config.alt_orography.is_file():
            raise FileNotFoundError(
                f"Neither orography file '{config.orography}' nor the "
                f"alternative orography file '{config.alt_orography}' exist."
            )
        logger.info(
            "stage 1 alt-orography fallback (regridding to %s, caching to %s)",
            config.orography_grid,
            config.orography,
        )
        alt_bytes = config.alt_orography.read_bytes()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "stage 1 alt-orography %s exists; reading %d bytes; "
                "regridding via mir_ops.interpolate",
                config.alt_orography,
                len(alt_bytes),
            )
        regridded = mir_ops.interpolate(
            alt_bytes, grid=config.orography_grid, method="grid-box-average"
        )
        config.orography.write_bytes(regridded)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "stage 1 writing %d bytes back to cache %s",
                len(regridded),
                config.orography,
            )
        return regridded

    raise FileNotFoundError(
        f"orography file '{config.orography}' does not exist; "
        f"pass --alt-orography to fall back to an alternative orography "
        f"input, which will be regridded to --orography-grid"
    )


def _stage_conservative_to_eres(orog_5km: bytes, config: SSOGenerateConfig) -> bytes:
    return mir_ops.interpolate(
        orog_5km, grid=config.effective_resolution, method="grid-box-average"
    )


def _stage_bilinear_back_to_orography_grid(
    orog_egrid: bytes, config: SSOGenerateConfig
) -> bytes:
    return mir_ops.interpolate(
        orog_egrid, grid=config.orography_grid, method="structured-bilinear"
    )


def _stage_difference_and_squared_difference(
    orog_5km: bytes,
    orog_egrid_og: bytes,
    config: SSOGenerateConfig,
) -> Tuple[bytes, bytes]:
    """Stage 4 — produce ``orog_egrid_diff`` and ``orog_egrid_diff_sq``."""
    orog_5km_arr, template = _decode(orog_5km)
    orog_egrid_og_arr, _ = _decode(orog_egrid_og)

    diff = orog_5km_arr - orog_egrid_og_arr
    _log_array("stage 4 diff", diff)
    diff = _maybe_roundtrip(diff, template, config, stage="4 diff")

    diff_sq = (orog_5km_arr - orog_egrid_og_arr) ** 2
    _log_array("stage 4 diff_sq", diff_sq)
    diff_sq = _maybe_roundtrip(diff_sq, template, config, stage="4 diff_sq")

    diff_bytes = _encode_on_template(diff, template)
    diff_sq_bytes = _encode_on_template(diff_sq, template)
    return diff_bytes, diff_sq_bytes


def _stage_gradient(orog_egrid_diff_bytes: bytes) -> Tuple[bytes, bytes]:
    return mir_ops.gradient(orog_egrid_diff_bytes, poles_missing_values=True)


def _stage_gradient_products(
    gradx_bytes: bytes,
    grady_bytes: bytes,
    config: SSOGenerateConfig,
) -> Tuple[bytes, bytes, bytes]:
    gradx, template = _decode(gradx_bytes)
    grady, _ = _decode(grady_bytes)

    gxx = gradx * gradx
    _log_array("stage 6 gxx", gxx)
    gxx = _maybe_roundtrip(gxx, template, config, stage="6 gxx")

    gyy = grady * grady
    _log_array("stage 6 gyy", gyy)
    gyy = _maybe_roundtrip(gyy, template, config, stage="6 gyy")

    gxy = gradx * grady
    _log_array("stage 6 gxy", gxy)
    gxy = _maybe_roundtrip(gxy, template, config, stage="6 gxy")

    return (
        _encode_on_template(gxx, template),
        _encode_on_template(gyy, template),
        _encode_on_template(gxy, template),
    )


def _stage_aggregate_to_eres(
    fields: Tuple[bytes, bytes, bytes, bytes], config: SSOGenerateConfig
) -> Tuple[bytes, bytes, bytes, bytes]:
    aggregated = tuple(
        mir_ops.interpolate(
            payload,
            grid=config.effective_resolution,
            method="grid-box-average",
        )
        for payload in fields
    )
    return aggregated  # type: ignore[return-value]


def _stage_aggregate_to_target(
    fields: Tuple[bytes, bytes, bytes, bytes], config: SSOGenerateConfig
) -> Tuple[bytes, bytes, bytes, bytes]:
    aggregated = tuple(
        mir_ops.interpolate(payload, grid=config.target_grid, method="grid-box-average")
        for payload in fields
    )
    return aggregated  # type: ignore[return-value]


def _stage_stdgwd(
    orog_mgrid_diff_sq: bytes, land_mask: np.ndarray, config: SSOGenerateConfig
) -> bytes:
    diff_sq, template = _decode(orog_mgrid_diff_sq)
    stdgwd = np.sqrt(diff_sq) * land_mask
    _log_array("stage 9.1 stdgwd", stdgwd)
    stdgwd = _maybe_roundtrip(stdgwd, template, config, stage="9.1 stdgwd")
    return encode_grib(
        stdgwd,
        template,
        metadata=_output_metadata("sdor", config),
    )


def _stage_klmlprime_lsm(
    gradxx_bytes: bytes,
    gradyy_bytes: bytes,
    gradxy_bytes: bytes,
    land_mask_bytes: bytes,
    land_mask: np.ndarray,
    config: SSOGenerateConfig,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bytes,
]:
    """Compute the K, L, M, Lprime bundle."""
    gradxx, template = _decode(gradxx_bytes)
    gradyy, _ = _decode(gradyy_bytes)
    gradxy, _ = _decode(gradxy_bytes)

    K = 0.5 * (gradxx + gradyy)
    L = 0.5 * (gradxx - gradyy)
    M = gradxy
    Lprime = np.sqrt((0.5 * (gradxx - gradyy)) ** 2 + gradxy**2)

    _log_array("stage 9.2.a K", K)
    _log_array("stage 9.2.a L", L)
    _log_array("stage 9.2.a M", M)
    _log_array("stage 9.2.a Lprime", Lprime)
    K = _maybe_roundtrip(K, template, config, stage="9.2.a K")
    L = _maybe_roundtrip(L, template, config, stage="9.2.a L")
    M = _maybe_roundtrip(M, template, config, stage="9.2.a M")
    Lprime = _maybe_roundtrip(Lprime, template, config, stage="9.2.a Lprime")
    _ = land_mask_bytes  # kept in signature for symmetry / future hooks

    return K, L, M, Lprime, land_mask, template


def _stage_slogwd(
    K: np.ndarray,
    Lprime: np.ndarray,
    land_mask: np.ndarray,
    template: bytes,
    config: SSOGenerateConfig,
) -> bytes:
    slogwd = np.sqrt(K + Lprime) * land_mask
    _log_array("stage 9.2.b slogwd", slogwd)
    slogwd = _maybe_roundtrip(slogwd, template, config, stage="9.2.b slogwd")
    return encode_grib(
        slogwd,
        template,
        metadata=_output_metadata("slor", config),
    )


def _stage_isogwd(
    K: np.ndarray,
    Lprime: np.ndarray,
    land_mask: np.ndarray,
    template: bytes,
    config: SSOGenerateConfig,
) -> bytes:
    """Stage 9.2.c — anisotropy.

    Faithful Python port of the ksh formula 10:

        sqrt( ((f1 - f4) * K_Lprime_gt_0)
              / ((f1 + f4) * K_Lprime_gt_epsilon + 0.00000001) ) * land_mask
    """
    epsilon = 0.00000001
    k_lprime_gt_0 = ((K - Lprime) > 0).astype(np.float64)
    k_lprime_gt_eps = ((K + Lprime) > epsilon).astype(np.float64)
    numerator = (K - Lprime) * k_lprime_gt_0
    denominator = (K + Lprime) * k_lprime_gt_eps + epsilon
    isogwd = np.sqrt(numerator / denominator) * land_mask
    _log_array("stage 9.2.c isogwd", isogwd)
    isogwd = _maybe_roundtrip(isogwd, template, config, stage="9.2.c isogwd")
    return encode_grib(
        isogwd,
        template,
        metadata=_output_metadata("isor", config),
    )


def _stage_anggwd(
    L: np.ndarray,
    M: np.ndarray,
    land_mask: np.ndarray,
    template: bytes,
    config: SSOGenerateConfig,
) -> bytes:
    """Stage 9.2.d — orientation. ``atan2(y=M, x=L)`` — argument order matters."""
    anggwd = 0.5 * np.arctan2(M, L) * land_mask
    _log_array("stage 9.2.d anggwd", anggwd)
    anggwd = _maybe_roundtrip(anggwd, template, config, stage="9.2.d anggwd")
    return encode_grib(
        anggwd,
        template,
        metadata=_output_metadata("anor", config),
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def compute_sso(config: SSOGenerateConfig) -> dict[str, bytes]:
    """Run the ten-stage SSO pipeline end-to-end.

    The pipeline implements the three-grid operational model:

    * ``source``      — grid the input orography arrives on
                        (auto-detected from the GRIB's ``gridName``).
    * ``orography_grid`` — high-resolution working grid where SSO
                        statistics are computed (operationally N2000;
                        tests use N256).
    * ``effective_resolution`` — coarse aggregation grid derived from
                        the model grid.
    * ``target_grid`` — final IFS model grid on which the four outputs
                        are written.

    Returns
    -------
    dict[str, bytes]
        Keys ``stdgwd``, ``slogwd``, ``anggwd``, ``isogwd``; each value
        is a single-message GRIB byte buffer with ``packingType=grid_simple``
        and the corresponding ``shortName``.
    """
    with _stage("1 source → orography grid"):
        orog_5km_bytes = _stage_source_to_orography_grid(config)

    with _stage("2 orography grid → effective resolution"):
        orog_egrid_bytes = _stage_conservative_to_eres(orog_5km_bytes, config)
    _write_intermediate("orog_egrid", orog_egrid_bytes, config)

    with _stage("3 effective resolution → orography grid (bilinear)"):
        orog_egrid_og_bytes = _stage_bilinear_back_to_orography_grid(
            orog_egrid_bytes, config
        )
    _write_intermediate(
        f"orog_egrid_{config.orography_grid}", orog_egrid_og_bytes, config
    )

    with _stage("4 diff and diff²"):
        orog_egrid_diff_bytes, orog_egrid_diff_sq_bytes = (
            _stage_difference_and_squared_difference(
                orog_5km_bytes, orog_egrid_og_bytes, config
            )
        )
    _write_intermediate("orog_egrid_diff", orog_egrid_diff_bytes, config)

    with _stage("5 scalar gradient"):
        gradx_bytes, grady_bytes = _stage_gradient(orog_egrid_diff_bytes)
    if config.dump_intermediates:
        _write_intermediate("orog_egrid_diff_grad", gradx_bytes + grady_bytes, config)

    with _stage("6 gradient products"):
        gradx_sq_bytes, grady_sq_bytes, gradxy_bytes = _stage_gradient_products(
            gradx_bytes, grady_bytes, config
        )
    _write_intermediate("orog_egrid_diff_gradx_sq", gradx_sq_bytes, config)
    _write_intermediate("orog_egrid_diff_grady_sq", grady_sq_bytes, config)
    _write_intermediate("orog_egrid_diff_gradxy", gradxy_bytes, config)

    with _stage("7 aggregate to effective resolution"):
        eres_bundle = _stage_aggregate_to_eres(
            (
                orog_egrid_diff_sq_bytes,
                gradx_sq_bytes,
                grady_sq_bytes,
                gradxy_bytes,
            ),
            config,
        )
    (
        orog_eff_diff_sq_bytes,
        orog_eff_diff_gradx_sq_bytes,
        orog_eff_diff_grady_sq_bytes,
        orog_eff_diff_gradxy_bytes,
    ) = eres_bundle
    _write_intermediate("orog_eff_diff_sq", orog_eff_diff_sq_bytes, config)
    _write_intermediate("orog_eff_diff_gradx_sq", orog_eff_diff_gradx_sq_bytes, config)
    _write_intermediate("orog_eff_diff_grady_sq", orog_eff_diff_grady_sq_bytes, config)
    _write_intermediate("orog_eff_diff_gradxy", orog_eff_diff_gradxy_bytes, config)

    with _stage("8 aggregate to target grid"):
        target_bundle = _stage_aggregate_to_target(eres_bundle, config)
    (
        orog_mgrid_diff_sq_bytes,
        orog_mgrid_diff_gradx_sq_bytes,
        orog_mgrid_diff_grady_sq_bytes,
        orog_mgrid_diff_gradxy_bytes,
    ) = target_bundle
    _write_intermediate("orog_mgrid_diff_sq", orog_mgrid_diff_sq_bytes, config)
    _write_intermediate(
        "orog_mgrid_diff_gradx_sq", orog_mgrid_diff_gradx_sq_bytes, config
    )
    _write_intermediate(
        "orog_mgrid_diff_grady_sq", orog_mgrid_diff_grady_sq_bytes, config
    )
    _write_intermediate("orog_mgrid_diff_gradxy", orog_mgrid_diff_gradxy_bytes, config)

    with _stage("9.1 stdgwd"):
        land_mask_bytes = config.land_mask.read_bytes()
        land_mask, _ = decode_grib(land_mask_bytes)
        stdgwd_bytes = _stage_stdgwd(orog_mgrid_diff_sq_bytes, land_mask, config)

    with _stage("9.2 KLMLprime / slogwd / isogwd / anggwd"):
        K, L, M, Lprime, lsm, mgrid_template = _stage_klmlprime_lsm(
            orog_mgrid_diff_gradx_sq_bytes,
            orog_mgrid_diff_grady_sq_bytes,
            orog_mgrid_diff_gradxy_bytes,
            land_mask_bytes,
            land_mask,
            config,
        )
        if config.dump_intermediates:
            # K, L, M, Lprime, land_mask — order load-bearing (matches ksh's
            # --variables=K;L;M;Lprime;land_mask, which formula 10 references
            # positionally as f1..f5).
            bundle = b"".join(
                encode_grib(arr, mgrid_template) for arr in (K, L, M, Lprime, lsm)
            )
            _write_intermediate("KLMLprime_lsm", bundle, config)

        slogwd_bytes = _stage_slogwd(K, Lprime, lsm, mgrid_template, config)
        isogwd_bytes = _stage_isogwd(K, Lprime, lsm, mgrid_template, config)
        anggwd_bytes = _stage_anggwd(L, M, lsm, mgrid_template, config)

    return {
        "stdgwd": stdgwd_bytes,
        "slogwd": slogwd_bytes,
        "anggwd": anggwd_bytes,
        "isogwd": isogwd_bytes,
    }


def generate(config: SSOGenerateConfig) -> dict[str, bytes]:
    """Product entry point.

    Resolves the config (inferring model/effective grids from
    ``target_grid`` when the operator did not pin them explicitly) and
    hands off to :func:`compute_sso`. Errors from Stage 1's file /
    grid checks propagate as ``FileNotFoundError`` / ``ValueError`` —
    the dispatcher (``pproc.climate.generate.__main__``) converts them
    into clean non-zero exits with the ``pproc-climate-fields
    sso: error: ...`` prefix.
    """
    resolved = config.resolve()
    return compute_sso(resolved)
