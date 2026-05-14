# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Sub-grid scale orography (SSO) pipeline.

A faithful Python port of ``generate_subgrid_orography_sso.ksh`` (see
``.weave/learnings/sso-migration.md`` for the ten-stage decomposition and
the eleven mir-compute formulae). The pipeline takes a 5 km source
orography on the working grid and a land mask on the target grid and
produces four GRIB byte buffers:

================  =========  ================
Result key        shortName  Description
================  =========  ================
``stdgwd``        ``sdor``   Std. dev. of subgrid orography
``slogwd``        ``slor``   Std. dev. of slope
``anggwd``        ``anor``   Orientation
``isogwd``        ``isor``   Anisotropy
================  =========  ================

The pipeline supports two opt-in modes via :class:`SSOConfig`:

* ``grib_roundtrip=True`` — encode/decode every numpy result through GRIB
  before it leaves a stage. This reproduces the per-step GRIB quantisation
  that the original ksh script applied via the ``cat`` + ``mir-compute``
  pattern, delivering value-array bit-identity with the reference data.
* ``dump_intermediates=True`` — write the 16 named intermediates to
  ``config.output_dir`` using their canonical ksh filenames. Filenames are
  hard-coded (no user-supplied path component), so path traversal is not a
  concern.

Sequential execution; no implicit threading at the pipeline layer.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pproc.climate import mir_ops
from pproc.climate.sso.config import SSOConfig
from pproc.common.io import decode_grib, encode_grib

__all__ = ["compute_sso"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decode(grib_bytes: bytes) -> Tuple[np.ndarray, bytes]:
    """Decode the first message in ``grib_bytes`` and return ``(values,
    template_bytes)``.

    The template bytes are kept in their wire form so they can be reused as
    an :func:`encode_grib` template without losing the section-1/section-4
    metadata. Returning ``grib_bytes`` directly is fine: ``encode_grib``
    only reads the first message of a multi-message buffer.
    """
    values, _ = decode_grib(grib_bytes)
    return values, grib_bytes


def _maybe_roundtrip(
    values: np.ndarray, template: bytes, config: SSOConfig
) -> np.ndarray:
    """Apply per-step GRIB quantisation when ``grib_roundtrip`` is on.

    The original ksh script writes every intermediate to disk as GRIB and
    reads it back for the next stage; mir-compute internally re-encodes
    its output to the input's packing. Reproducing this behaviour at every
    numpy step is the only way to land bit-identity at the values array
    against the reference outputs, hence the watch-item in the K1 sign-off.
    """
    if not config.grib_roundtrip:
        return values
    encoded = encode_grib(values, template)
    decoded, _ = decode_grib(encoded)
    return decoded


def _write_intermediate(name: str, payload: bytes, config: SSOConfig) -> None:
    """Persist a named intermediate when ``dump_intermediates`` is on.

    ``name`` is hard-coded at every call site (one per row of the
    "Stage → intermediate file mapping" table in the migration learnings),
    so the joined path cannot escape ``output_dir`` via traversal. The
    output directory is created on demand to keep the test fixture's
    ``tmp_path`` integration clean.
    """
    if not config.dump_intermediates:
        return
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / name).write_bytes(payload)


def _encode_on_template(values: np.ndarray, template: bytes) -> bytes:
    """Encode ``values`` against ``template``'s wire bytes.

    Thin convenience wrapper that keeps the call sites tidy.
    """
    return encode_grib(values, template)


def _output_metadata(short_name: str, config: SSOConfig) -> dict:
    """Build the output metadata dict, optionally pinning bitsPerValue.

    When ``config.bits_per_value`` is ``None`` (the default), the returned
    dict omits the ``bitsPerValue`` key entirely so that eccodes inherits
    or defaults the value from the packing in use (``grid_simple``). When
    set, the value is added so the user gets the precision they asked for.
    """
    metadata: dict = {"shortName": short_name, "packingType": "grid_simple"}
    if config.bits_per_value is not None:
        metadata["bitsPerValue"] = config.bits_per_value
    return metadata


# ---------------------------------------------------------------------------
# Stage helpers — one per row of the ten-stage table
# ---------------------------------------------------------------------------


def _stage_conservative_to_n2000(config: SSOConfig) -> bytes:
    """Stage 1 — orography on the working (N2000/N256) grid.

    If ``config.orography`` already exists on disk, this is a no-op pass
    through. Otherwise the canonical raw ``source_orography`` is
    interpolated via ``grid-box-average`` to ``output_grid``.

    The conditional matches the ksh script:

        if [[ ! -f ${inFile} ]] ; then
           run_mir --grid=$OUT_RES --interpolation=grid-box-average \\
               ${inFile_alt} $fileName
        fi
    """
    if config.orography.is_file():
        return config.orography.read_bytes()
    if config.source_orography is None:
        raise FileNotFoundError(
            f"orography {config.orography!s} does not exist and no "
            "source_orography fallback is configured"
        )
    raw = config.source_orography.read_bytes()
    return mir_ops.interpolate(raw, grid=config.output_grid, method="grid-box-average")


def _stage_conservative_to_eres(orog_5km: bytes, config: SSOConfig) -> bytes:
    """Stage 2 — orography aggregated to the effective resolution (eres)."""
    return mir_ops.interpolate(
        orog_5km, grid=config.effective_resolution, method="grid-box-average"
    )


def _stage_bilinear_back_to_n2000(orog_egrid: bytes, config: SSOConfig) -> bytes:
    """Stage 3 — orog_egrid bilinearly interpolated back to N2000."""
    return mir_ops.interpolate(
        orog_egrid, grid=config.output_grid, method="structured-bilinear"
    )


def _stage_difference_and_squared_difference(
    orog_5km: bytes,
    orog_egrid_n2000: bytes,
    config: SSOConfig,
) -> Tuple[bytes, bytes]:
    """Stage 4 — produce ``orog_egrid_diff`` and ``orog_egrid_diff_sq``.

    Equivalent to the ksh ``cat $inFile orog_egrid_N2000 > tmp`` followed
    by the two ``mir-compute`` calls. We keep the template aligned with
    ``orog_5km`` (the first concatenated message), matching the ksh order.
    """
    orog_5km_arr, template = _decode(orog_5km)
    orog_egrid_n2000_arr, _ = _decode(orog_egrid_n2000)

    diff = orog_5km_arr - orog_egrid_n2000_arr
    diff = _maybe_roundtrip(diff, template, config)

    diff_sq = (orog_5km_arr - orog_egrid_n2000_arr) ** 2
    diff_sq = _maybe_roundtrip(diff_sq, template, config)

    diff_bytes = _encode_on_template(diff, template)
    diff_sq_bytes = _encode_on_template(diff_sq, template)
    return diff_bytes, diff_sq_bytes


def _stage_gradient(orog_egrid_diff_bytes: bytes) -> Tuple[bytes, bytes]:
    """Stage 5 — scalar gradient (∂h/∂x, ∂h/∂y) of ``orog_egrid_diff``."""
    return mir_ops.gradient(orog_egrid_diff_bytes, poles_missing_values=True)


def _stage_gradient_products(
    gradx_bytes: bytes,
    grady_bytes: bytes,
    config: SSOConfig,
) -> Tuple[bytes, bytes, bytes]:
    """Stage 6 — three pointwise products of the gradient components."""
    gradx, template = _decode(gradx_bytes)
    grady, _ = _decode(grady_bytes)

    gxx = gradx * gradx
    gxx = _maybe_roundtrip(gxx, template, config)

    gyy = grady * grady
    gyy = _maybe_roundtrip(gyy, template, config)

    gxy = gradx * grady
    gxy = _maybe_roundtrip(gxy, template, config)

    return (
        _encode_on_template(gxx, template),
        _encode_on_template(gyy, template),
        _encode_on_template(gxy, template),
    )


def _stage_aggregate_to_eres(
    fields: Tuple[bytes, bytes, bytes, bytes], config: SSOConfig
) -> Tuple[bytes, bytes, bytes, bytes]:
    """Stage 7 — conservative aggregation of four N256 fields to eres."""
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
    fields: Tuple[bytes, bytes, bytes, bytes], config: SSOConfig
) -> Tuple[bytes, bytes, bytes, bytes]:
    """Stage 8 — conservative aggregation of four eres fields to the target."""
    aggregated = tuple(
        mir_ops.interpolate(payload, grid=config.target_grid, method="grid-box-average")
        for payload in fields
    )
    return aggregated  # type: ignore[return-value]


def _stage_stdgwd(
    orog_mgrid_diff_sq: bytes, land_mask: np.ndarray, config: SSOConfig
) -> bytes:
    """Stage 9.1 — ``stdgwd = sqrt(orog_mgrid_diff_sq) * land_mask``."""
    diff_sq, template = _decode(orog_mgrid_diff_sq)
    stdgwd = np.sqrt(diff_sq) * land_mask
    stdgwd = _maybe_roundtrip(stdgwd, template, config)
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
    config: SSOConfig,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bytes,
]:
    """Stage 9.2.a — compute the K, L, M, Lprime bundle.

    Returns the five numpy arrays in the canonical bundle order
    ``(K, L, M, Lprime, land_mask)`` plus the template bytes (taken from
    the first input, ``gradxx``) for downstream encoding.

    Bundle order is load-bearing: the ksh formula 10 references ``f1`` and
    ``f4`` positionally, which only works if the bundle is K-first /
    Lprime-fourth (matching ``--variables=K;L;M;Lprime;land_mask``).
    """
    gradxx, template = _decode(gradxx_bytes)
    gradyy, _ = _decode(gradyy_bytes)
    gradxy, _ = _decode(gradxy_bytes)

    # Five sub-formulae from the ksh's --formula= line 237.
    K = 0.5 * (gradxx + gradyy)
    L = 0.5 * (gradxx - gradyy)
    M = gradxy
    Lprime = np.sqrt((0.5 * (gradxx - gradyy)) ** 2 + gradxy**2)

    K = _maybe_roundtrip(K, template, config)
    L = _maybe_roundtrip(L, template, config)
    M = _maybe_roundtrip(M, template, config)
    Lprime = _maybe_roundtrip(Lprime, template, config)
    # land_mask is already on disk as GRIB; no further roundtrip is needed.
    _ = land_mask_bytes  # kept in signature for symmetry / future hooks

    return K, L, M, Lprime, land_mask, template


def _stage_slogwd(
    K: np.ndarray,
    Lprime: np.ndarray,
    land_mask: np.ndarray,
    template: bytes,
    config: SSOConfig,
) -> bytes:
    """Stage 9.2.b — ``slogwd = sqrt(K + Lprime) * land_mask``."""
    slogwd = np.sqrt(K + Lprime) * land_mask
    slogwd = _maybe_roundtrip(slogwd, template, config)
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
    config: SSOConfig,
) -> bytes:
    """Stage 9.2.c — anisotropy.

    Faithful Python port of the ksh formula 10:

        sqrt( ((f1 - f4) * K_Lprime_gt_0)
              / ((f1 + f4) * K_Lprime_gt_epsilon + 0.00000001) ) * land_mask

    where ``f1=K``, ``f4=Lprime``, ``K_Lprime_gt_0 = (K-Lprime) > 0`` and
    ``K_Lprime_gt_epsilon = (K+Lprime) > 0.00000001``. The epsilon literal
    ``0.00000001`` (= 1e-8) is reproduced verbatim — the K1 watch-items
    flag this as a D-F1 candidate if outputs drift.
    """
    epsilon = 0.00000001
    k_lprime_gt_0 = ((K - Lprime) > 0).astype(np.float64)
    k_lprime_gt_eps = ((K + Lprime) > epsilon).astype(np.float64)
    numerator = (K - Lprime) * k_lprime_gt_0
    denominator = (K + Lprime) * k_lprime_gt_eps + epsilon
    isogwd = np.sqrt(numerator / denominator) * land_mask
    isogwd = _maybe_roundtrip(isogwd, template, config)
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
    config: SSOConfig,
) -> bytes:
    """Stage 9.2.d — orientation.

    The ksh formula 11 is ``0.5 * atan2(M, L) * land_mask``. Argument
    order matters: ``atan2(y=M, x=L)``, NOT ``atan2(L, M)``. K1 watch-items
    flag this as a D-F1 candidate if outputs drift.
    """
    anggwd = 0.5 * np.arctan2(M, L) * land_mask
    anggwd = _maybe_roundtrip(anggwd, template, config)
    return encode_grib(
        anggwd,
        template,
        metadata=_output_metadata("anor", config),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_sso(config: SSOConfig) -> dict[str, bytes]:
    """Run the ten-stage SSO pipeline end-to-end.

    Parameters
    ----------
    config:
        A resolved :class:`SSOConfig` instance. ``output_grid``,
        ``effective_resolution``, ``model_grid_type`` and
        ``model_resolution`` are expected to be filled in (call
        ``config.resolve()`` first).

    Returns
    -------
    dict[str, bytes]
        Keys ``stdgwd``, ``slogwd``, ``anggwd``, ``isogwd``; each value is
        a single-message GRIB byte buffer with ``packingType=grid_simple``
        and the corresponding ``shortName``.
    """
    # ------- Stage 1: source → working grid (or pass through) -----------
    orog_5km_bytes = _stage_conservative_to_n2000(config)

    # ------- Stage 2: → effective resolution ---------------------------
    orog_egrid_bytes = _stage_conservative_to_eres(orog_5km_bytes, config)
    _write_intermediate("orog_egrid", orog_egrid_bytes, config)

    # ------- Stage 3: ← bilinear back to N2000 -------------------------
    orog_egrid_n2000_bytes = _stage_bilinear_back_to_n2000(orog_egrid_bytes, config)
    _write_intermediate("orog_egrid_N2000", orog_egrid_n2000_bytes, config)

    # ------- Stage 4: difference + squared difference ------------------
    orog_egrid_diff_bytes, orog_egrid_diff_sq_bytes = (
        _stage_difference_and_squared_difference(
            orog_5km_bytes, orog_egrid_n2000_bytes, config
        )
    )
    _write_intermediate("orog_egrid_diff", orog_egrid_diff_bytes, config)

    # ------- Stage 5: scalar gradient ----------------------------------
    gradx_bytes, grady_bytes = _stage_gradient(orog_egrid_diff_bytes)
    if config.dump_intermediates:
        _write_intermediate("orog_egrid_diff_grad", gradx_bytes + grady_bytes, config)

    # ------- Stage 6: gradient products --------------------------------
    gradx_sq_bytes, grady_sq_bytes, gradxy_bytes = _stage_gradient_products(
        gradx_bytes, grady_bytes, config
    )
    _write_intermediate("orog_egrid_diff_gradx_sq", gradx_sq_bytes, config)
    _write_intermediate("orog_egrid_diff_grady_sq", grady_sq_bytes, config)
    _write_intermediate("orog_egrid_diff_gradxy", gradxy_bytes, config)

    # ------- Stage 7: aggregate to eres --------------------------------
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

    # ------- Stage 8: aggregate to target ------------------------------
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

    # ------- Stage 9.1: stdgwd ----------------------------------------
    land_mask_bytes = config.land_mask.read_bytes()
    land_mask, _ = decode_grib(land_mask_bytes)
    stdgwd_bytes = _stage_stdgwd(orog_mgrid_diff_sq_bytes, land_mask, config)

    # ------- Stage 9.2.a: K, L, M, Lprime bundle ----------------------
    K, L, M, Lprime, lsm, mgrid_template = _stage_klmlprime_lsm(
        orog_mgrid_diff_gradx_sq_bytes,
        orog_mgrid_diff_grady_sq_bytes,
        orog_mgrid_diff_gradxy_bytes,
        land_mask_bytes,
        land_mask,
        config,
    )
    if config.dump_intermediates:
        # Concatenate K, L, M, Lprime, land_mask in that order to match
        # ``--variables=K;L;M;Lprime;land_mask`` in the ksh — formula 10
        # references f1/f4 positionally, so this order is load-bearing.
        bundle = b"".join(
            encode_grib(arr, mgrid_template) for arr in (K, L, M, Lprime, lsm)
        )
        _write_intermediate("KLMLprime_lsm", bundle, config)

    # ------- Stage 9.2.b–d: slogwd, isogwd, anggwd --------------------
    slogwd_bytes = _stage_slogwd(K, Lprime, lsm, mgrid_template, config)
    isogwd_bytes = _stage_isogwd(K, Lprime, lsm, mgrid_template, config)
    anggwd_bytes = _stage_anggwd(L, M, lsm, mgrid_template, config)

    return {
        "stdgwd": stdgwd_bytes,
        "slogwd": slogwd_bytes,
        "anggwd": anggwd_bytes,
        "isogwd": isogwd_bytes,
    }
