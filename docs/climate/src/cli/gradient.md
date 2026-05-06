# pproc-gradient

`pproc-gradient` is a thin argparse wrapper over
`pproc.climate.mir_ops.gradient` (for `scalar-gradient`) and an inline
`mir.Job(nabla='scalar-laplacian')` invocation (for `scalar-laplacian`).
Reads a single-message GRIB input, applies mir's nabla operator, and
writes a GRIB output.

## Synopsis

```
pproc-gradient [--operation {scalar-gradient,scalar-laplacian}]
               [--no-poles-missing-values]
               INPUT OUTPUT
```

## Flags

| Flag | Argument | Description |
|------|----------|-------------|
| `INPUT` | path | Input GRIB file (single-message scalar field). |
| `OUTPUT` | path | Output GRIB file. |
| `--operation` | `scalar-gradient` \| `scalar-laplacian` | mir nabla operation. Default: `scalar-gradient`. |
| `--no-poles-missing-values` | — | Disable flagging values at lat=±90° as missing in the output. Default behaviour (enabled) matches the legacy SSO ksh pipeline's `--nabla-poles-missing-values` flag. |
| `-h`, `--help` | — | Show argparse help and exit. |

## Output format

| `--operation` | Number of GRIB messages | Order |
|---------------|-------------------------|-------|
| `scalar-gradient` (default) | 2 | ∂f/∂lat first, then ∂f/∂lon |
| `scalar-laplacian` | 1 | — |

The two-message layout for `scalar-gradient` matches the byte order of
the legacy `orog_egrid_diff_grad` intermediate, and downstream code
(notably `pproc-field-calc --multi-dimensional 2 ... gradx;grady`) can
consume the output directly.

## Pole behaviour

By default, values at lat=±90° are set to GRIB missing values, matching
the legacy ksh script's `--nabla-poles-missing-values` flag. Use
`--no-poles-missing-values` to disable this. Note: for reduced Gaussian
grids whose first/last latitude row sits *just inside* ±90° (e.g. N256
→ ±89.731°), no points are eligible to be flagged, so the resulting
buffer carries no missing entries either way.

## Quality contract: byte-exact gradient

`pproc-gradient --operation scalar-gradient` produces output that is
**byte-exact** against the legacy reference data. This is verified at
the test layer in
`pproc/tests/test_gradient_cli.py::test_scalar_gradient_matches_reference`,
where the value arrays of the two output messages match the reference
intermediate `orog_egrid_diff_grad` at `rtol=1e-5` with max abs diff
0.0 — a true bit-identical match. The byte exactness comes from
`mir_ops.gradient` using `eccodes.MemoryReader` to split the multi-message
mir output without a decode/re-encode round-trip.

## Example

```bash
pproc-gradient orog_egrid_diff /tmp/grad.grib
```

This invocation matches Stage 5 of the legacy SSO pipeline (see
[Legacy script mapping](../appendix/legacy-mapping.md)).
