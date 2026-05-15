# pproc-sso

`pproc-sso` is the monolithic CLI driving the sub-grid orography
pipeline end-to-end. It reads an orography GRIB and a land-mask GRIB
on the target grid, runs the ten-stage SSO computation
(`pproc.climate.sso.pipeline.compute_sso`), and writes the four output
GRIB files (`stdgwd`, `slogwd`, `anggwd`, `isogwd`) into
`--output-dir`.

## Synopsis

```
pproc-sso --orography PATH --land-mask PATH --target-grid GRID
          --orography-grid GRID
          [--alt-orography PATH]
          [--model-grid-type TYPE] [--model-resolution RES]
          [--effective-resolution GRID]
          [--output-dir DIR]
          [--grib-roundtrip] [--dump-intermediates]
          [--bits-per-value N]
          [--config FILE]
```

## Three-grid model

The SSO pipeline reasons about three distinct grids plus the
effective-resolution aggregation grid:

| Role | Where it comes from | Operational value |
|------|----------------------|-------------------|
| `source` | The `gridName` of the input `--orography` GRIB (auto-detected). | Whatever the upstream producer emits (often O256 raw IFS output). |
| `orography_grid` | `--orography-grid` flag (required). | `N2000` (≈ 5 km); the legacy ksh script hardcoded this at lines 106 and 128. Tests use `N256`. |
| `effective_resolution` (eres) | Derived from the model grid via Unit C. | E.g. `N48` for an `O80` model. |
| `target_grid` | `--target-grid` flag (required). | The IFS model grid the four outputs land on. |

The legacy ksh script conflated `orography_grid` with `target_grid`
through a single `$OUT_RES` variable. `pproc-sso` keeps them separate
so the operator can compute SSO statistics on a high-resolution
working grid and aggregate to any target grid they need.

## Stage 1 grid handling

Stage 1 reads the input GRIB's `gridName` via `decode_grib` and
compares it to `--orography-grid`. `--orography` is treated as
**authoritative**: the operator declares "this file is on
`--orography-grid`". If the file's actual grid matches, the bytes
pass through unchanged. If it differs, that is a configuration error
and Stage 1 raises `ValueError` — the pipeline does **not** silently
regrid `--orography`.

If you have an orography on a different grid than `--orography-grid`,
pass it as `--alt-orography` and the pipeline will regrid and cache
it for you (see
[Cached vs alternative orography input](#cached-vs-alternative-orography-input)).

## Cached vs alternative orography input

`--orography` is the preferred input: a pre-staged GRIB on the
working grid. The pipeline supports a two-file fallback workflow
that mirrors the legacy ksh's `orog_5km` / `orog` (`$inFile` /
`$inFile_alt`) pattern:

1. **Fast path / pass through** — `--orography` exists and its
   `gridName` already equals `--orography-grid`. The bytes pass
   straight through Stage 1.
2. **Grid mismatch (configuration error)** — `--orography` exists
   but is on a different grid than `--orography-grid`. Stage 1
   raises `ValueError`; the CLI surfaces this as a clean non-zero
   exit. The on-disk file is **not** modified. To fix: either
   replace the file with one already on `--orography-grid`, or
   move this file to `--alt-orography` to have it regridded (case
   3 below). The regrid step happens only via `--alt-orography`,
   never silently from `--orography`.
3. **Fallback with cache writeback** — `--orography` does *not* exist
   and `--alt-orography` is supplied. The alternative is regridded to
   `--orography-grid` via `grid-box-average` and the result is
   written to the `--orography` path. Subsequent runs find the
   cached file and take the fast path.

This caching behaviour is part of the contract: it mirrors the legacy
ksh's `cp $fileName ${XDATA_IFS}/$fileName` step (line 109 of
`generate_subgrid_orography_sso.ksh`). Operators are expected to
pre-create the parent directory of `--orography`; if it does not
exist, the underlying `write_bytes` call propagates the OS
`FileNotFoundError`. Concurrent runs writing the same cache file are
out of scope (matching the ksh).

Cache writeback only happens on case 3.

### Error behaviour

* `--orography` exists but is on a different grid than
  `--orography-grid` → `ValueError`:
  `orography file '<path>' is on grid '<input-grid>' but
  --orography-grid is '<orography-grid>'; supply an orography on
  '<orography-grid>', or move this file to --alt-orography to have
  it regridded.`
* `--orography` missing, `--alt-orography` not supplied →
  `FileNotFoundError`:
  `orography file '<path>' does not exist; pass --alt-orography to
  fall back to an alternative orography input, which will be
  regridded to --orography-grid`.
* `--orography` missing, `--alt-orography` supplied but also missing →
  `FileNotFoundError`:
  `Neither orography file '<orography-path>' nor the alternative
  orography file '<alt-path>' exist.`

All three messages surface as clean non-zero exits from the CLI
(`pproc-sso: error: <message>`), without a Python traceback.

## Flags

| Flag | Argument | Description |
|------|----------|-------------|
| `--orography` | `PATH` | Path to orography GRIB; treated as authoritative for `--orography-grid`. If the file's actual grid differs, Stage 1 raises a configuration error — pass the file as `--alt-orography` instead to have it regridded. Required (CLI or YAML). |
| `--alt-orography` | `PATH` | Alternative orography input, used as a fallback when `--orography` does not exist on disk; the result is regridded to `--orography-grid` and cached at the `--orography` path. See [Cached vs alternative orography input](#cached-vs-alternative-orography-input). |
| `--land-mask` | `PATH` | Path to land-mask GRIB on the target grid. Required (CLI or YAML). |
| `--target-grid` | `GRID` | Final target/output grid spec, e.g. `N256` or `O1280`. Required (CLI or YAML). |
| `--orography-grid` | `GRID` | High-resolution working grid where SSO statistics are computed. Operationally `N2000`; tests use `N256`. Required (CLI or YAML). |
| `--model-grid-type` | `TYPE` | Model grid family code (`O`, `N`, `F`). Auto-inferred from `--target-grid` when omitted. |
| `--model-resolution` | `RES` | Model nominal resolution (integer, e.g. 80). Auto-inferred from `--target-grid` when omitted. |
| `--effective-resolution` | `GRID` | Override the auto-computed effective-resolution grid spec (e.g. `N48`). Defaults to the value derived from the model grid (see [Effective resolution mapping](#effective-resolution-mapping)). |
| `--output-dir` | `DIR` | Directory in which to write the four output files (default: `.`). Created on demand. |
| `--grib-roundtrip` | — | Encode/decode every numpy intermediate through GRIB to reproduce the per-step quantisation of the original ksh script. |
| `--dump-intermediates` | — | Write the sixteen named intermediate GRIB files to `--output-dir` in addition to the four final outputs. |
| `--bits-per-value` | `N` | Set the GRIB `bitsPerValue` on the four output fields. When omitted, `bitsPerValue` is not written by pproc — eccodes uses its own default for the packing (24 for `grid_simple`). Use `32` to match the legacy ksh script's output precision. |
| `--config` | `FILE` | Optional YAML config file. Keys must match `SSOConfig` field names (snake_case). CLI arguments override YAML values. |
| `-h`, `--help` | — | Show argparse help and exit. |

## Pipeline overview

`pproc-sso` runs the canonical ten-stage decomposition documented in
[Legacy script mapping](../appendix/legacy-mapping.md). The stage names
are stable and are also the keys used internally in
`pproc.climate.sso.pipeline`:

1. `source_to_orography_grid` — interpolate (or pass through) source orography onto `orography_grid`.
2. `conservative_to_eres` — interpolate to effective resolution.
3. `bilinear_back_to_orography_grid` — bilinear back to `orography_grid`.
4. `compute_diff_and_diff_sq` — difference and squared difference.
5. `compute_gradient` — `mir.Job(nabla='scalar-gradient')`.
6. `square_gradients` — element-wise products of the gradient components.
7. `aggregate_to_eres` — grid-box-average back to effective resolution.
8. `aggregate_to_target` — grid-box-average to target grid.
9. `compute_sso_outputs` — compute `stdgwd`, `slogwd`, `isogwd`, `anggwd`.
10. `archive` — write final outputs to `--output-dir`.

## Output fields

The four final fields, with their canonical short names and `paramId`
slugs, are:

| File | shortName | Meaning |
|------|-----------|---------|
| `stdgwd` | `sdor` | Standard deviation of subgrid orography |
| `slogwd` | `slor` | Standard deviation of slope |
| `anggwd` | `anor` | Anisotropy orientation |
| `isogwd` | `isor` | Anisotropy ratio |

Mapping: `stdgwd → sdor`, `slogwd → slor`, `anggwd → anor`, `isogwd → isor`.

## Operational toggles

### `--grib-roundtrip`

Encode and decode the numpy intermediates through GRIB after every
arithmetic step (Stages 4, 6, 9.1, 9.2.a–d), reproducing the per-step
GRIB quantisation of the legacy ksh script. This narrows the value-array
drift against the reference outputs but cannot close the float32 vs
float64 arithmetic gap; see [D-F1 tolerance posture](#d-f1-tolerance-posture)
below.

### `--bits-per-value`

Sets the GRIB `bitsPerValue` on the four output fields (`stdgwd`,
`slogwd`, `anggwd`, `isogwd`). When omitted, pproc does **not** write
`bitsPerValue` at all — eccodes inherits/defaults it from the packing
in use, which is `24` for `grid_simple`.

Why the legacy outputs land at `bitsPerValue=32` while pproc's default
is `24`: the legacy ksh chain runs `grid_simple` end-to-end via
`mir-compute`, which carries `bitsPerValue=32` forward through every
re-encode. The pproc chain stays in `grid_ieee` for the numpy
intermediates and only switches to `grid_simple` at the final encode,
so absent an explicit knob the eccodes default applies. Pass
`--bits-per-value 32` to reproduce the legacy precision exactly.

### `--dump-intermediates`

Writes the sixteen named intermediate GRIB files to `--output-dir`. The
filenames are hard-coded with one exception: the bilinear-back-to-working-grid
intermediate is parameterised on `--orography-grid` and lands at
`orog_egrid_<orography_grid>` (e.g. `orog_egrid_N2000` operationally,
`orog_egrid_N256` in tests). The full list:

```
orog_egrid                       orog_egrid_<orography_grid>   orog_egrid_diff
orog_egrid_diff_grad             orog_egrid_diff_gradx_sq      orog_egrid_diff_grady_sq
orog_egrid_diff_gradxy           orog_eff_diff_sq              orog_eff_diff_gradx_sq
orog_eff_diff_grady_sq           orog_eff_diff_gradxy          orog_mgrid_diff_sq
orog_mgrid_diff_gradx_sq         orog_mgrid_diff_grady_sq      orog_mgrid_diff_gradxy
KLMLprime_lsm
```

Two of these files contain multiple GRIB messages: `orog_egrid_diff_grad`
holds the two scalar-gradient components (∂f/∂lat, ∂f/∂lon), and
`KLMLprime_lsm` holds the five fields used by the SSO output stage in
order `K, L, M, Lprime, land_mask`.

## YAML config

`--config FILE` loads a YAML document via `yaml.safe_load` and merges
its fields into the parsed argparse namespace. CLI arguments take
precedence: any value the user passes on the command line overrides the
corresponding YAML field. Boolean flags can only be promoted from
`False` to `True` by YAML (the user cannot toggle them off via YAML
once enabled on the CLI).

The recognised YAML keys are exactly the snake-case `SSOConfig` field
names:

```yaml
orography: data/input/ifs/orog_5km
alt_orography: data/input/ifs/orog   # optional fallback; see workflow above
land_mask: data/input/ifs/land_mask
target_grid: N256
model_grid_type: O
model_resolution: 80
orography_grid: N2000
effective_resolution: N48
output_dir: ./out
grib_roundtrip: false
dump_intermediates: true
bits_per_value: 32
```

Unknown keys raise an error rather than being silently ignored.

## Effective resolution mapping

The effective resolution grid (`MIR_ERES_SET` in the legacy script) is
derived from `model_grid_type` and `model_resolution` per the table
documented in `pproc.climate.sso.effective_resolution`. For octahedral
(`model_grid_type=O`):

```
ERES = model_resolution / 2          # integer division
ERES = ERES - (ERES % 2)             # round DOWN to even
```

Special cases (operational, not derivable from the formula):

| `ERES` | Effective grid |
|--------|----------------|
| 40     | `N48`          |
| 100    | `N128`         |
| 1000   | `N1024`        |
| else   | `N${ERES}`     |

For non-octahedral grid types the effective grid equals the target
grid. Worked example: `model_grid_type=O, model_resolution=80` →
`ERES=40` → `N48`. See `pproc.climate.sso.effective_resolution` for the
full lookup logic and `appendix/legacy-mapping.md` for the back-pointer
to the ksh source.

## D-F1 tolerance posture

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

A longer-form analysis (the float32-vs-float64 root cause, the
`binaryScaleFactor=-21` GRIB grid_simple quantum, the path forward) is
in [Tolerance policy (D-F1)](../appendix/tolerance-policy.md).

## Example

```bash
pproc-sso \
  --orography data/input/ifs/orog_5km \
  --land-mask data/input/ifs/land_mask \
  --target-grid N256 \
  --model-grid-type O \
  --model-resolution 80 \
  --orography-grid N256 \
  --output-dir ./out
```

This matches the legacy test run (`GTYPE_SET=O, ORES=80,
MIR_GTYPE_SET=N256, MIR_ERES_SET=N48`) and produces the four final
files `out/stdgwd`, `out/slogwd`, `out/anggwd`, `out/isogwd`.
Operationally, set `--orography-grid N2000` instead, matching the
N2000 working grid hardcoded by the legacy script.
