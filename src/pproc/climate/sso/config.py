"""Pydantic configuration model for the SSO pipeline.

Mirrors the environment-variable surface of
``generate_subgrid_orography_sso.ksh`` (see the env-var table in
``.weave/learnings/sso-migration.md``). Tool-config knobs (``MIR_VERSION``,
``MIR_COMPUTE_VERSION``) are deliberately omitted -- those belong to the
runtime/PATH layer, not the pipeline contract.

The model is **not frozen**, but ``resolve()`` follows immutable semantics:
it never mutates ``self`` or the caller's input dict. Instead it returns a
``model_copy(update=...)`` of self with the auto-inferred fields filled in.
This makes idempotency a natural consequence of equality on the dumped
fields rather than a hand-rolled "already resolved" flag.

Three-grid operational model
----------------------------
The SSO pipeline reasons about three distinct grids:

* ``source``: the grid the input orography arrives on (often O256 raw IFS
  output). Read off the GRIB; not a configuration field.
* ``orography_grid``: the high-resolution working grid where SSO
  statistics are computed (operationally N2000 ≈ 5 km; in tests we use
  N256 to keep reference fixtures small). Required input.
* ``effective_resolution`` (``eres``): the coarse aggregation grid
  derived from the model grid (e.g. N48 for an O80 model). Computed by
  ``resolve()`` via Unit C.
* ``target_grid``: the final IFS model grid the four outputs land on.
  Required input.

The legacy ksh script conflated ``orography_grid`` with ``target_grid``
through a single ``$OUT_RES`` variable, which broke as soon as ops tried
to compute SSO for a target grid different from the working grid.
``SSOConfig`` keeps them separate.

Inference precedence in ``resolve()``:

1. If the caller supplied **both** ``model_grid_type`` and
   ``model_resolution`` explicitly, those win.
2. Otherwise both are inferred from ``target_grid`` via
   :func:`pproc.climate.sso.effective_resolution.infer_grid_params`.
3. ``effective_resolution`` is always recomputed from the resolved
   ``model_grid_type``/``model_resolution`` via
   :func:`pproc.climate.sso.effective_resolution.compute_effective_resolution`.

The ksh test-run config is the canonical reason this precedence matters:
``MIR_GTYPE_SET=N256`` (output) but ``GTYPE_SET=O`` and ``ORES=80`` (model),
which yields ``MIR_ERES_SET=N48`` -- i.e. the eres derives from the
*model* grid, not from the *output* grid.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pproc.climate.sso.effective_resolution import (
    compute_effective_resolution,
    infer_grid_params,
)

__all__ = ["SSOConfig"]


class SSOConfig(BaseModel):
    """Configuration for the sub-grid scale orography (SSO) pipeline.

    Field-to-env-var mapping (see ``.weave/learnings/sso-migration.md``):

    =========================  ==========================================
    SSOConfig field            ksh env var / source
    =========================  ==========================================
    ``orography``              ``$inFile``
    ``alt_orography``          ``$inFile_alt``
    ``land_mask``              ``$maskFile``
    ``target_grid``            ``$MIR_GTYPE_SET``
    ``model_grid_type``        ``$GTYPE_SET``
    ``model_resolution``       ``$ORES``
    ``orography_grid``         hardcoded ``N2000`` at lines 106 and 128 of
                               ``generate_subgrid_orography_sso.ksh`` (no
                               variable in the legacy script)
    ``effective_resolution``   ``$MIR_ERES_SET`` (derived)
    ``output_dir``             ``$OUTPUT_DIR``
    ``bits_per_value``         (no env-var counterpart)
    =========================  ==========================================

    ``bits_per_value`` has no legacy env-var: the ksh script never set
    ``bitsPerValue`` explicitly. The ``bitsPerValue=32`` observed on its
    outputs is inherited from the mir-compute encoding chain
    (``grid_simple`` end-to-end), not from a user-supplied knob.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Inputs ---------------------------------------------------------
    orography: Path = Field(
        ...,
        description="Source orography GRIB file (ksh: $inFile).",
    )
    alt_orography: Path | None = Field(
        default=None,
        description=(
            "Alternative orography input. Used as a fallback when "
            "``orography`` does not exist on disk: the alternative is "
            "regridded to ``orography_grid`` and the result is cached at "
            "the ``orography`` path so subsequent runs hit the fast path. "
            "Matches the legacy ksh script's ``inFile_alt`` variable."
        ),
    )
    land_mask: Path = Field(
        ...,
        description="Land mask GRIB on target grid (ksh: $maskFile).",
    )

    # --- Grid configuration --------------------------------------------
    target_grid: str = Field(
        ...,
        min_length=1,
        description="Target output grid (ksh: $MIR_GTYPE_SET, e.g. 'N256').",
    )
    model_grid_type: str = Field(
        default="",
        description=(
            "Model grid family code (ksh: $GTYPE_SET, e.g. 'O' or 'N'). "
            "Auto-inferred from ``target_grid`` when both this and "
            "``model_resolution`` are left at their defaults."
        ),
    )
    model_resolution: int = Field(
        default=0,
        ge=0,
        description=(
            "Model nominal resolution (ksh: $ORES, e.g. 80). Auto-inferred "
            "from ``target_grid`` when both this and ``model_grid_type`` "
            "are left at their defaults."
        ),
    )
    orography_grid: str = Field(
        ...,
        min_length=1,
        description=(
            "High-resolution working grid where SSO statistics are computed. "
            "Operationally ``N2000`` (≈ 5 km, hardcoded at lines 106 and 128 "
            "of the legacy ``generate_subgrid_orography_sso.ksh``); in tests "
            "we use ``N256`` to keep reference fixtures small. Required: the "
            "user must always supply this explicitly."
        ),
    )
    effective_resolution: str = Field(
        default="",
        description=(
            "Effective-resolution grid (ksh: $MIR_ERES_SET). Always "
            "computed by ``resolve()`` from the model grid via Unit C."
        ),
    )

    # --- Output ---------------------------------------------------------
    output_dir: Path = Field(
        default=Path("."),
        description="Directory for the four final outputs (ksh: $OUTPUT_DIR).",
    )

    # --- Pipeline behaviour toggles ------------------------------------
    grib_roundtrip: bool = Field(
        default=False,
        description=(
            "Encode/decode GRIB after every numpy step to reproduce the "
            "per-step quantization of the original ksh script."
        ),
    )
    dump_intermediates: bool = Field(
        default=False,
        description="Write the 16 named intermediate files to disk for debugging.",
    )
    bits_per_value: int | None = Field(
        default=None,
        gt=0,
        description=(
            "If set, override the GRIB ``bitsPerValue`` on the four output "
            "fields (stdgwd, slogwd, anggwd, isogwd). When ``None`` (default), "
            "``bitsPerValue`` is not written: eccodes inherits/defaults the "
            "value from the packing in use (24 for ``grid_simple``). Set to "
            "32 to match the legacy ksh script's output precision."
        ),
    )

    # ------------------------------------------------------------------
    def resolve(self) -> "SSOConfig":
        """Return a copy with all auto-inferred fields populated.

        Idempotent: ``cfg.resolve().resolve() == cfg.resolve()``. Does not
        mutate ``self`` or the dict the caller passed to ``__init__``.
        """
        # Inference precedence: if the caller supplied BOTH the grid type
        # and the resolution, take them as given. Otherwise infer both
        # from target_grid -- a partial override (only one of the two
        # provided) is treated as "not explicit enough" and falls back to
        # inference. This matches the ksh script, which only ever sets
        # GTYPE_SET and ORES as a pair.
        explicit_model = bool(self.model_grid_type) and self.model_resolution > 0

        if explicit_model:
            grid_type = self.model_grid_type
            resolution = self.model_resolution
        else:
            grid_type, resolution = infer_grid_params(self.target_grid)

        effective_resolution = compute_effective_resolution(grid_type, resolution)

        return self.model_copy(
            update={
                "model_grid_type": grid_type,
                "model_resolution": resolution,
                "effective_resolution": effective_resolution,
            }
        )
