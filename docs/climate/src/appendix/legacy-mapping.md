# Legacy script mapping

Source of truth: `generate_subgrid_orography_sso.ksh` at the workspace
root. Line numbers below cite that file. The decomposition follows the
"Stage ordering" and "mir-compute formulae" sections of
`.weave/learnings/sso-migration.md`.

## Ten-stage mapping

| # | Stage (snake_case ID) | ksh comment | ksh tool | Python counterpart |
|---|---|---|---|---|
| 1 | `conservative_to_n2000` | "Interpolate source data to N2000 (5 km) if input file does not exist" | `run_mir --grid=$OUT_RES --interpolation=grid-box-average` | `pproc.climate.mir_ops.interpolate(src, grid=OUT_RES, method="grid-box-average")` inside `pproc.climate.sso.pipeline.compute_sso` (Stage 1; conditional on input-file presence). |
| 2 | `conservative_to_eres` | "Interpolate N2000 (5 km) to effective resolution" | `run_mir --grid=${MIR_ERES_SET} --interpolation=grid-box-average` | `interpolate(orog_5km, grid=eres, method="grid-box-average")` |
| 3 | `bilinear_back_to_n2000` | "Interpolate effective resolution back to N2000 with bilinear interpolation" | `run_mir --grid=$OUT_RES --interpolation=structured-bilinear` | `interpolate(orog_egrid, grid=OUT_RES, method="structured-bilinear")` |
| 4 | `compute_diff_and_diff_sq` | "Take difference and squared difference between N2000 (input from step 1) and N2000 from step 2" | `cat $inFile orog_egrid_N2000 > tmp; mir_compute ...` ×2 | `evaluate_formula` ×2 in `compute_sso` (in-memory bundle, no `cat`). |
| 5 | `compute_gradient` | "Compute derivatives of difference" | `run_mir --nabla=scalar-gradient --nabla-poles-missing-values` | `pproc.climate.mir_ops.gradient(orog_egrid_diff, poles_missing_values=True)` |
| 6 | `square_gradients` | "Square/multiply derivatives from step 4" | `mir_compute` ×3 (multiDimensional 2) | `evaluate_formula` ×3 |
| 7 | `aggregate_to_eres` | "Aggregate results of steps 3 and 5 to effective resolution" | `run_mir --grid=${MIR_ERES_SET} --interpolation=grid-box-average` ×4 | `interpolate(..., grid=eres, method="grid-box-average")` ×4 |
| 8 | `aggregate_to_target` | "Aggregate results of step 6 to target resolution" | `run_mir --grid=${MIR_GTYPE_SET} --interpolation=grid-box-average` ×4 | `interpolate(..., grid=target, method="grid-box-average")` ×4 |
| 9 | `compute_sso_outputs` | "Compute standard deviation of subgrid orography (stdgwd), …" | `mir_compute` ×6 + `grib_set` ×4 | `evaluate_formula` ×6 + metadata application via `encode_grib(metadata=...)` |
| 10 | `archive` | "Archive fields to ECFS" | shell `mv` ×4 | `(config.output_dir / name).write_bytes(results[name])` for each of `stdgwd`, `slogwd`, `anggwd`, `isogwd`. |

## Per-formula mapping

The legacy script issues eleven `mir_compute` invocations producing
fifteen sub-formulae once the multi-output rows (`;`-separated
expressions) are exploded. The table below shows the verbatim ksh
formula, the equivalent Python expression executed inside
`pproc.climate.field_calc.evaluate_formula`, and the equivalent
`pproc-field-calc` CLI invocation.

In the CLI column, `tmp` denotes a single GRIB stream produced by
in-memory concatenation of the input messages (the legacy script writes
this with shell `cat`; the Python pipeline keeps it in `BytesIO`).

### Stage 4 (lines 138–144)

| # | ksh `mir_compute` formula | Python expression | CLI |
|---|---|---|---|
| 1 | `orog_N2000 - orog_egrid_N2000` | `evaluate_formula("orog_N2000 - orog_egrid_N2000", {"orog_N2000": a, "orog_egrid_N2000": b})` | `pproc-field-calc --variables "orog_N2000;orog_egrid_N2000" --formula "orog_N2000 - orog_egrid_N2000" --multi-dimensional 2 tmp orog_egrid_diff` |
| 2 | `(orog_N2000 - orog_egrid_N2000)^2` | `evaluate_formula("(orog_N2000 - orog_egrid_N2000)^2", {...})` | `pproc-field-calc --variables "orog_N2000;orog_egrid_N2000" --formula "(orog_N2000 - orog_egrid_N2000)^2" --multi-dimensional 2 tmp orog_egrid_diff_sq` |

### Stage 6 (lines 161–171)

| # | ksh formula | Python | CLI |
|---|---|---|---|
| 3 | `gradx * gradx` | `evaluate_formula("gradx * gradx", {"gradx": gx, "grady": gy})` | `pproc-field-calc --variables "gradx;grady" --formula "gradx * gradx" --multi-dimensional 2 orog_egrid_diff_grad orog_egrid_diff_gradx_sq` |
| 4 | `grady * grady` | `evaluate_formula("grady * grady", {...})` | `pproc-field-calc --variables "gradx;grady" --formula "grady * grady" --multi-dimensional 2 orog_egrid_diff_grad orog_egrid_diff_grady_sq` |
| 5 | `gradx * grady` | `evaluate_formula("gradx * grady", {...})` | `pproc-field-calc --variables "gradx;grady" --formula "gradx * grady" --multi-dimensional 2 orog_egrid_diff_grad orog_egrid_diff_gradxy` |

### Stage 9.1 (lines 220–223)

| # | ksh formula | Python | CLI |
|---|---|---|---|
| 6 | `sqrt(orog_mgrid_diff_sq) * land_mask` | `evaluate_formula("sqrt(orog_mgrid_diff_sq) * land_mask", {"orog_mgrid_diff_sq": x, "land_mask": m})` | `pproc-field-calc --variables "orog_mgrid_diff_sq;land_mask" --formula "sqrt(orog_mgrid_diff_sq) * land_mask" --metadata shortName=sdor packingType=grid_simple --multi-dimensional 2 tmp stdgwd` |

### Stage 9.2 — bundle `KLMLprime_lsm` (line 237; multi-output, 5 sub-formulae)

The single ksh `--formula` string at line 237 is semicolon-separated
and produces five GRIB messages.

| # | ksh sub-formula | Python | Output column in `KLMLprime_lsm` |
|---|---|---|---|
| 7a | `0.5*(gradxx+gradyy)` | `evaluate_formula("0.5*(gradxx+gradyy)", {...})` | K |
| 7b | `0.5*(gradxx-gradyy)` | `evaluate_formula("0.5*(gradxx-gradyy)", {...})` | L |
| 7c | `gradxy` | `evaluate_formula("gradxy", {...})` | M |
| 7d | `sqrt((0.5*(gradxx-gradyy))^2+(gradxy)^2)` | `evaluate_formula("sqrt((0.5*(gradxx-gradyy))^2+(gradxy)^2)", {...})` | Lprime |
| 7e | `land_mask` | `evaluate_formula("land_mask", {...})` | land_mask |

CLI:

```bash
pproc-field-calc --variables "gradxx;gradyy;gradxy;land_mask" \
                 --formula "0.5*(gradxx+gradyy); 0.5*(gradxx-gradyy); gradxy; sqrt((0.5*(gradxx-gradyy))^2+(gradxy)^2); land_mask" \
                 --multi-dimensional 4 tmp KLMLprime_lsm
```

### Stage 9.2 — slogwd (lines 248–251)

| # | ksh formula | Python | CLI |
|---|---|---|---|
| 8 | `sqrt(K + Lprime) * land_mask` | `evaluate_formula("sqrt(K + Lprime) * land_mask", {"K": k, "L": l, "M": m, "Lprime": lp, "land_mask": lm})` | `pproc-field-calc --variables "K;L;M;Lprime;land_mask" --formula "sqrt(K + Lprime) * land_mask" --metadata shortName=slor packingType=grid_simple --multi-dimensional 5 KLMLprime_lsm slogwd` |

### Stage 9.2 — limits bundle (line 257; multi-output, 2 sub-formulae)

| # | ksh sub-formula | Python | Output column in `limits` |
|---|---|---|---|
| 9a | `(K - Lprime) > 0` | `evaluate_formula("(K - Lprime) > 0", {...})` | K_Lprime_gt_0 |
| 9b | `(K + Lprime) > 0.00000001` | `evaluate_formula("(K + Lprime) > 0.00000001", {...})` | K_Lprime_gt_epsilon |

CLI:

```bash
pproc-field-calc --variables "K;L;M;Lprime;land_mask" \
                 --formula "(K - Lprime) > 0; (K + Lprime) > 0.00000001" \
                 --multi-dimensional 5 KLMLprime_lsm limits
```

Both sub-formulae return `float64` 0/1 fields (comparisons return
floats by design — see [pproc-field-calc](../cli/field-calc.md)).

### Stage 9.2 — isogwd (lines 263–266)

| # | ksh formula | Python | CLI |
|---|---|---|---|
| 10 | `sqrt( ((f1 - f4) * K_Lprime_gt_0) / ((f1 + f4) * K_Lprime_gt_epsilon + 0.00000001) ) * land_mask` | Equivalent Python with `f1 → K`, `f4 → Lprime` (positional aliases): `evaluate_formula("sqrt( ((K - Lprime) * K_Lprime_gt_0) / ((K + Lprime) * K_Lprime_gt_epsilon + 0.00000001) ) * land_mask", {...})` | `pproc-field-calc --variables "K;L;M;Lprime;land_mask;K_Lprime_gt_0;K_Lprime_gt_epsilon" --formula "sqrt( ((f1 - f4) * K_Lprime_gt_0) / ((f1 + f4) * K_Lprime_gt_epsilon + 0.00000001) ) * land_mask" --metadata shortName=isor packingType=grid_simple --multi-dimensional 7 KLMLprime_lsm_lim isogwd` |

The input file `KLMLprime_lsm_lim` is itself an in-memory concatenation
of `KLMLprime_lsm` (5 messages) and `limits` (2 messages); the bundle
order is load-bearing because formula 10 references `f1` and `f4`
positionally.

### Stage 9.2 — anggwd (lines 270–273)

| # | ksh formula | Python | CLI |
|---|---|---|---|
| 11 | `0.5 * atan2(M, L) * land_mask` | `evaluate_formula("0.5 * atan2(M, L) * land_mask", {...})` | `pproc-field-calc --variables "K;L;M;Lprime;land_mask" --formula "0.5 * atan2(M, L) * land_mask" --metadata shortName=anor packingType=grid_simple --multi-dimensional 5 KLMLprime_lsm anggwd` |

The argument order to `atan2` is `M, L` (not `L, M`); this matches
`numpy.arctan2(M, L)` semantics inside `evaluate_formula`.

## grib_set → metadata application

Each of the four `grib_set -s shortName=...,packingType=grid_simple -r
<tmp> <output>` invocations in the legacy script (lines 223, 251, 266,
273) is replaced by `--metadata shortName=... packingType=grid_simple`
on the corresponding `pproc-field-calc` invocation, or by a
`metadata={"shortName": ..., "packingType": "grid_simple"}` argument to
`encode_grib` inside `compute_sso`.

| Final output | shortName | packingType | ksh line |
|--------------|-----------|-------------|----------|
| `stdgwd` | `sdor` | `grid_simple` | 223 |
| `slogwd` | `slor` | `grid_simple` | 251 |
| `isogwd` | `isor` | `grid_simple` | 266 |
| `anggwd` | `anor` | `grid_simple` | 273 |

## Effective resolution mapping

The `MIR_ERES_SET` variable is computed at lines 73–90 of the ksh
script. The table is reproduced in [pproc-sso §
Effective resolution mapping](../cli/sso.md#effective-resolution-mapping)
and implemented in `pproc.climate.sso.effective_resolution`. The
operational specials (`40 → N48`, `100 → N128`, `1000 → N1024`) are
table-documented rather than derived — they are not expressible from
the `ERES = ORES/2; round-down-to-even` formula alone.
