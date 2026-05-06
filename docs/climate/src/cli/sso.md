# pproc-sso

`pproc-sso` is the monolithic CLI driving the sub-grid orography
pipeline end-to-end. It reads a 5 km orography GRIB and a land-mask GRIB
on the target grid, runs the ten-stage SSO computation
(`pproc.climate.sso.pipeline.compute_sso`), and writes the four output
GRIB files (`stdgwd`, `slogwd`, `anggwd`, `isogwd`) into
`--output-dir`.

## Synopsis

```
pproc-sso [--orography PATH] [--land-mask PATH] [--target-grid GRID]
          [--source-orography PATH]
          [--model-grid-type TYPE] [--model-resolution RES]
          [--effective-resolution GRID] [--output-grid GRID]
          [--output-dir DIR]
          [--grib-roundtrip] [--dump-intermediates]
          [--config FILE]
```

## Flags

| Flag | Argument | Description |
|------|----------|-------------|
| `--orography` | `PATH` | Path to source orography GRIB on the working grid. Required (CLI or YAML). |
| `--land-mask` | `PATH` | Path to land-mask GRIB on the target grid. Required (CLI or YAML). |
| `--target-grid` | `GRID` | Target/output grid spec, e.g. `N256` or `O1280`. Required (CLI or YAML). |
| `--source-orography` | `PATH` | Optional fallback raw orography. Used by Stage 1 to (re)generate the working-grid orography when `--orography` does not exist. |
| `--model-grid-type` | `TYPE` | Model grid family code (`O`, `N`, `F`). Auto-inferred from `--target-grid` when omitted. |
| `--model-resolution` | `RES` | Model nominal resolution (integer, e.g. 80). Auto-inferred from `--target-grid` when omitted. |
| `--effective-resolution` | `GRID` | Override the auto-computed effective-resolution grid spec (e.g. `N48`). Defaults to the value derived from the model grid (see [Effective resolution mapping](#effective-resolution-mapping)). |
| `--output-grid` | `GRID` | Output grid (`OUT_RES`) for working stages. Defaults to `--target-grid`. |
| `--output-dir` | `DIR` | Directory in which to write the four output files (default: `.`). Created on demand. |
| `--grib-roundtrip` | — | Encode/decode every numpy intermediate through GRIB to reproduce the per-step quantisation of the original ksh script. |
| `--dump-intermediates` | — | Write the sixteen named intermediate GRIB files to `--output-dir` in addition to the four final outputs. |
| `--config` | `FILE` | Optional YAML config file. Keys must match `SSOConfig` field names (snake_case). CLI arguments override YAML values. |
| `-h`, `--help` | — | Show argparse help and exit. |

## Pipeline overview

`pproc-sso` runs the canonical ten-stage decomposition documented in
[Legacy script mapping](../appendix/legacy-mapping.md). The stage names
are stable and are also the keys used internally in
`pproc.climate.sso.pipeline`:

1. `conservative_to_n2000` — interpolate source orography to N2000 (5 km).
2. `conservative_to_eres` — interpolate to effective resolution.
3. `bilinear_back_to_n2000` — bilinear back to N2000.
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

### `--dump-intermediates`

Writes the sixteen named intermediate GRIB files to `--output-dir`. The
filenames are hard-coded (no user-controlled component) and match the
intermediate filenames the legacy ksh script wrote to the working
directory:

```
orog_egrid                     orog_egrid_N2000              orog_egrid_diff
orog_egrid_diff_grad           orog_egrid_diff_gradx_sq      orog_egrid_diff_grady_sq
orog_egrid_diff_gradxy         orog_eff_diff_sq              orog_eff_diff_gradx_sq
orog_eff_diff_grady_sq         orog_eff_diff_gradxy          orog_mgrid_diff_sq
orog_mgrid_diff_gradx_sq       orog_mgrid_diff_grady_sq      orog_mgrid_diff_gradxy
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
land_mask: data/input/ifs/land_mask
target_grid: N256
model_grid_type: O
model_resolution: 80
output_grid: N256
effective_resolution: N48
source_orography: data/input/255_4/orog
output_dir: ./out
grib_roundtrip: false
dump_intermediates: true
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
  --output-dir ./out
```

This matches the legacy test run (`GTYPE_SET=O, ORES=80, OUT_RES=N256,
MIR_GTYPE_SET=N256, MIR_ERES_SET=N48`) and produces the four final
files `out/stdgwd`, `out/slogwd`, `out/anggwd`, `out/isogwd`.
