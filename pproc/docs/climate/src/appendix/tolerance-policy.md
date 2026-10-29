# Tolerance policy (D-F1)

This page records the tolerance posture for the climate-fields
migration's reference-output match. The original decision was taken at
the K2 pre-CLI coordination gate for the SSO product; the same
float32-vs-float64 arithmetic-path pattern has since been observed on
two additional products (`albedo` and `orography-variance`), documented
below. The verbatim short-form note for SSO is reproduced in
[pproc-climate-fields sso § D-F1 tolerance
posture](../cli/sso.md#d-f1-tolerance-posture).

## D-F1 outcome — verbatim from the K2 sign-off (SSO)

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

The reference outputs were produced by the legacy ksh scripts via
`mir-compute`, which evaluates `sqrt`, `atan2`, and mixed
arithmetic operations in **float32** internally and stores
intermediates as `grid_simple` GRIB. The new Python pipeline evaluates
the same expressions in `pproc.formula.evaluate_formula`, which
dispatches to `numpy.sqrt`, `numpy.arctan2`, and friends — operating
in **float64** throughout. As a consequence, three of the twenty-seven
climate-field products cannot be reproduced from the reference
intermediates at bit-identical precision regardless of whether
`--grib-roundtrip` is on. The drift is purely arithmetic precision,
not algorithmic; the new outputs are arguably more accurate than the
reference.

For SSO, the drift was isolated by feeding the *reference*
`orog_mgrid_diff_sq` and `land_mask` directly into `np.sqrt(x) * mask`
and reproducing the same divergence — i.e. the gap is independent of
any pipeline-internal step. The diagnostic is in
`pproc/tests/climate/generate/test_pipeline.py` (module docstring
lines 24–67).

## The grid_simple quantum at `binaryScaleFactor=-21` (SSO)

`stdgwd` exhibits the largest absolute drift, ≈ 6.1 × 10⁻⁵. This is
exactly **128 × 2⁻²¹**: the GRIB1 `grid_simple` packing of the
reference field uses `binaryScaleFactor = -21`, giving a quantisation
step of 2⁻²¹ ≈ 4.77 × 10⁻⁷ per packed value. The float32 `sqrt(x) *
mask` result is then re-quantised at this resolution before being
written to disk; the float64 result, written through the same packing,
quantises to a *different* nearest representable value at points where
the float32 and float64 sqrt outputs straddle a quantum boundary. The
peak drift is therefore bounded by a small multiple of the quantum.

The other three SSO fields exhibit drifts at the float32 precision
floor:

- `slogwd`: max abs diff ≈ 9.3 × 10⁻¹⁰ (machine-epsilon scale; only
  fails `rtol` because some reference values are themselves ≈ 10⁻¹⁰).
- `isogwd`: max abs diff ≈ 3 × 10⁻⁸ (float32 precision near zero).
- `anggwd`: max abs diff ≈ 10⁻⁷ (float32 `atan2` precision near zero).

## Additional products with sub-packing D-F1 drift

Two other products exhibit the same class of drift when compared
against the legacy ksh reference outputs on the local test harness
(N48 target grid, 30-arc-second source data). Both remain below the
GRIB packing quantum used to store the reference; neither product
widens its own tolerance thresholds.

### `albedo` — 2 files, 1 pixel each

`pproc-climate-fields albedo` produces six monthly outputs per input
regime. Two of the twelve reference files differ at one pixel each,
where the float32 `sqrt` and boundary-condition arithmetic straddles
a `grid_simple` quantum boundary. All other pixels bit-match. Drift
magnitude is at the packing quantum for the affected month/regime
combinations.

### `orography-variance` — 1879 pixels of 13280

The formula `total_variance = mean_of_squares − mean²` is a
catastrophic-cancellation identity: when two large positive numbers
are subtracted to yield a small residual, the residual's precision is
bounded by the least-precise summand. `mir-compute` performs the
subtraction in float32; `pproc.formula.evaluate_formula` performs it
in float64. The two paths diverge on 1879 of the 13280 N48-target
grid points; all diffs remain below the `grid_simple` packing
precision used to store the reference. Bit-identical reproduction
would require inserting a float32 cast at the same subtraction point
`mir-compute` uses; see [Path forward](#path-forward).

## Path forward

If operations require bit-identical reproduction of the legacy outputs
for the three affected products (`sso`, `albedo`,
`orography-variance`), the right fix is to cast to **float32** at the
same points `mir-compute` does — namely the `sqrt` / `atan2` calls in
SSO Stage 9, the boundary-condition arithmetic in `albedo`, and the
`mean_of_squares − mean²` subtraction in `orography-variance` —
rather than weaken the test tolerances. Each is a localised pipeline
change inside the product's `generate()` function; none alter any CLI
argument surface and each can be added in a follow-up unit without
breaking compatibility.

None of the [`pproc-climate-fields`](../cli/climate-fields.md) product
CLIs expose `--rtol` / `--atol` flags. Tolerances belong to tests, not
to the runtime contract.

## Current resolution (SSO)

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

The SSO resolution is recorded at the **Pattern level** (i.e. accepted
at the coordination gate that authorises CLI dispatch), pending
operational sign-off. The K2 sign-off explicitly notes that user
confirmation is still outstanding.

For `albedo` and `orography-variance`, the D-F1 characterisation is
recorded at the Pattern level pending operational review of the
per-pixel drift magnitudes. Until then, the tolerance floors above
(for SSO) and the sub-packing-quantum bound (for the other two
products) are the authoritative reference-match contract for the
migration.
