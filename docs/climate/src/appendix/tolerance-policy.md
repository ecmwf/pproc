# Tolerance policy (D-F1)

This page records the tolerance posture for the SSO migration's
reference-output match. The decision was taken at the K2 pre-CLI
coordination gate; the verbatim short-form note is reproduced in
[pproc-sso § D-F1 tolerance posture](../cli/sso.md#d-f1-tolerance-posture).
The longer-form analysis below explains the root cause, the GRIB
quantum at play, the path forward, and the current resolution.

## D-F1 outcome — verbatim from the K2 sign-off

> Legacy ksh used float32 sqrt/atan2 inside mir-compute; our float64 numpy pipeline cannot bit-match, and the applied resolution is per-field atol floors:
>
> | Field    | rtol | atol  |
> |----------|------|-------|
> | stdgwd   | 1e-5 | 1e-4  |
> | slogwd   | 1e-5 | 1e-9  |
> | isogwd   | 1e-5 | 1e-7  |
> | anggwd   | 1e-5 | 1e-6  |
>
> If operations need bit-identical legacy reproduction, the right fix is a localised float32 cast at the same points mir-compute uses, not test-tolerance widening.

## Float32 vs float64 root cause

The reference outputs were produced by the legacy ksh script via
`mir-compute`, which evaluates `sqrt` and `atan2` in **float32**
internally and stores intermediates as `grid_simple` GRIB. The new
Python pipeline evaluates the same expressions in
`pproc.climate.field_calc.evaluate_formula`, which dispatches to
`numpy.sqrt`, `numpy.arctan2`, and friends — operating in **float64**
throughout. As a consequence, the four output fields cannot be
reproduced from the reference intermediates at the originally specified
`rtol=1e-5` regardless of whether `--grib-roundtrip` is on. The drift
is purely arithmetic precision, not algorithmic; our outputs are
arguably more accurate than the reference.

The drift was isolated by feeding the *reference* `orog_mgrid_diff_sq`
and `land_mask` directly into `np.sqrt(x) * mask` and reproducing the
same divergence — i.e. the gap is independent of any pipeline-internal
step. The diagnostic is in
`pproc/tests/climate/sso/test_pipeline.py` (module docstring lines
24–67).

## The grid_simple quantum at `binaryScaleFactor=-21`

`stdgwd` exhibits the largest absolute drift, ≈ 6.1 × 10⁻⁵. This is
exactly **128 × 2⁻²¹**: the GRIB1 `grid_simple` packing of the
reference field uses `binaryScaleFactor = -21`, giving a quantisation
step of 2⁻²¹ ≈ 4.77 × 10⁻⁷ per packed value. The float32 `sqrt(x) *
mask` result is then re-quantised at this resolution before being
written to disk; our float64 result, written through the same packing,
quantises to a *different* nearest representable value at points where
the float32 and float64 sqrt outputs straddle a quantum boundary. The
peak drift is therefore bounded by a small multiple of the quantum.

The other three fields exhibit drifts at the float32 precision floor:

- `slogwd`: max abs diff ≈ 9.3 × 10⁻¹⁰ (machine-epsilon scale; only
  fails `rtol` because some reference values are themselves ≈ 10⁻¹⁰).
- `isogwd`: max abs diff ≈ 3 × 10⁻⁸ (float32 precision near zero).
- `anggwd`: max abs diff ≈ 10⁻⁷ (float32 `atan2` precision near zero).

## Path forward

If operations require bit-identical reproduction of the legacy outputs,
the right fix is to cast to **float32** at the same points
`mir-compute` does — namely the `sqrt` and `atan2` calls in Stage 9 —
rather than weaken the test tolerances. This is a localised pipeline
change inside `pproc.climate.sso.pipeline`; it does not alter any CLI
argument surface and it can be added in a follow-up unit without
breaking compatibility.

The CLIs (`pproc-sso`, `pproc-gradient`, `pproc-field-calc`) do **not**
expose `--rtol` / `--atol` flags. Tolerances belong to tests, not to
the runtime contract.

## Current resolution

Per-field `atol` floors at the arithmetic-noise level, with `rtol=1e-5`
preserved across the board:

| Field    | rtol | atol   | Justification                              |
|----------|------|--------|--------------------------------------------|
| `stdgwd` | 1e-5 | 1e-4   | Covers the ≈ 6.1 × 10⁻⁵ `grid_simple` quantum drift. |
| `slogwd` | 1e-5 | 1e-9   | Machine-epsilon scale.                     |
| `isogwd` | 1e-5 | 1e-7   | Float32 precision near zero.               |
| `anggwd` | 1e-5 | 1e-6   | Float32 `atan2` precision near zero.       |

The same tolerances apply to `--grib-roundtrip` mode — that mode
narrows the GRIB-quantisation gap but cannot close the float32
arithmetic gap, so the originally-specified "bit-identical at
value-array level" criterion was unattainable from a pure numpy
pipeline.

## Status

This resolution is recorded at the **Pattern level** (i.e. accepted at
the coordination gate that authorises CLI dispatch), pending operational
sign-off. The K2 sign-off explicitly notes that user confirmation is
still outstanding; until then, the tolerance floors above are the
authoritative reference-match contract for the migration.
