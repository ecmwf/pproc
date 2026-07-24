# pproc-climate-fields

`pproc-climate-fields` is the unified CLI for generating IFS climate
fields. It replaces the twenty-seven mir/`mir-compute`-driven
[`ifs-scripts/clim/generate_*.ksh`](https://github.com/ecmwf-ifs/) scripts
with a single Python entry point that dispatches to one of twenty-seven
in-tree Python products. Each product owns its own algorithm, config
schema, and GRIB metadata; the tool itself does no per-field logic.

## Synopsis

```
pproc-climate-fields <field> [<field-flags>]
```

`<field>` is one of the twenty-seven names in the [Product catalogue](#product-catalogue).
The first argument is consumed by an argparse dispatcher; everything
after it is passed to a per-product Conflator app that parses the
field-specific flags and any `--config` YAML or `--set KEY=VAL`
overrides.

Discover the available fields:

```bash
pproc-climate-fields --help
```

Discover a specific field's flags:

```bash
pproc-climate-fields sso --help
pproc-climate-fields lake-depth --help
```

## Product catalogue

Alphabetical by field name. The "ksh source" column names the legacy
`ifs-scripts/clim/generate_*.ksh` script whose behaviour the product
reproduces bit-for-bit (subject to the [D-F1 tolerance
posture](../appendix/tolerance-policy.md) for the two float32/float64
drifts).

| Field | Description | ksh source |
|-------|-------------|------------|
| `albedo` | Monthly 6-component albedo on target grid (per-file, two regimes). | `generate_albedo.ksh` |
| `albedo-four-stream` | 4-component albedo + monthly OSM albedo on target grid. | `generate_albedo_four_stream.ksh` |
| `albedo-single-stream` | Monthly single-stream albedo on target grid (grid-box-average). | `generate_albedo_single_stream.ksh` |
| `aqua-planet` | Monthly aqua-planet climatology (7 lake fields; orographic correction). | `generate_aqua_planet.ksh` |
| `glacier-cover` | Glacier cover (cicecap) on target grid (grid-box-average interpolation). | `generate_glacier_cover.ksh` |
| `glacier-mask` | Binary glacier and glacier-free land masks (2 outputs). | `generate_glacier_mask.ksh` |
| `irrigation-cover` | Irrigation cover on target grid (grid-box-average interpolation). | `generate_irrigation_cover.ksh` |
| `lake-cover` | Inland water / lake cover on target grid (grid-box-average interpolation). | `generate_lake_cover.ksh` |
| `lake-depth` | Lake depth on the target grid via mode-filter + bilinear + clamp. | `generate_lake_depth.ksh` |
| `lake-mask` | Binary lake mask from lake-cover input via threshold ≥ 0.5. | `generate_lake_mask.ksh` |
| `land-cover` | Land cover (lsm) on target grid (grid-box-average interpolation). | `generate_land_cover.ksh` |
| `land-mask` | Binary land mask from land-cover input via threshold > 0.5. | `generate_land_mask.ksh` |
| `ocean-bathymetry` | Mean ocean bathymetry on target grid (grid-box-average interpolation). | `generate_ocean_bathymetry.ksh` |
| `ocean-mask` | Binary ocean mask = 1 − land_mask − lake_mask. | `generate_ocean_mask.ksh` |
| `oceanic-emissions` | Monthly DMS oceanic emissions on target grid (nearest-neighbour + ocean mask). | `generate_oceanic_emissions.ksh` |
| `orography` | Mean orography on target grid (grid-box-average interpolation). | `generate_orography.ksh` |
| `orography-variance` | Total orographic variance on target grid (law of total variance). | `generate_orography_variance.ksh` |
| `sea-surface` | Monthly SST + sea-ice on target grid (nearest-lsm + ocean mask). | `generate_sea_surface.ksh` |
| `soil-moisture` | ASCAT CDF matching parameters on target grid (nearest-lsm). | `generate_soil_moisture.ksh` |
| `soil-moisture-smos` | SMOS brightness-temperature bias correction on target grid (nearest-lsm). | `generate_soil_moisture_smos.ksh` |
| `soil-type` | Soil type on target grid (nearest-neighbour or mode-integral). | `generate_soil_type.ksh` |
| `soil-type-hwsd` | Soil type (HWSD, van Genuchten) on target grid. | `generate_soil_type_hwsd.ksh` |
| `sso` | Sub-grid orography: `stdgwd`/`slogwd`/`anggwd`/`isogwd` via the 10-stage pipeline. | `generate_subgrid_orography_sso.ksh` |
| `subgrid-orography-sdfor` | Std dev of orography 1–5 km on target grid (`sdfor`). | `generate_subgrid_orography_sdfor.ksh` |
| `urban-cover` | Urban cover on target grid (grid-box-average + land mask). | `generate_urban_cover.ksh` |
| `water-type` | Categorical water-type field (ocean + N water-bodies) on target grid. | `generate_water_type.ksh` |
| `wetland-cover` | Monthly wetland cover on target grid (12-month loop + land mask). | `generate_wetland_cover.ksh` |

Two legacy ksh scripts are **not** migrated to `pproc-climate-fields`
because they invoke non-mir external tools:

* `generate_spectral_orography.ksh` — MPI-parallel Fortran `gptosp.exe`
  via `run_parallel`.
* `generate_vegetation.ksh` — Python + on-the-fly Cython build
  (`agg_esacci_lc.py`, `agg_cgls.py`, `disagg_cgls.py`).

Both remain callable as their original ksh scripts. If they are ever
brought under `pproc-climate-fields` the pattern will be the same as
above: one product module, one field name, one `generate()` function
that owns the algorithm and the metadata.

## Shared configuration

Every product's config subclasses
`pproc.climate.generate.config.BaseGenerateConfig` and inherits the
following flags:

| Flag | Description |
|------|-------------|
| `--target-grid GRID` | Final output grid, e.g. `N256` or `O1280`. |
| `--bits-per-value N` | Optional GRIB `bitsPerValue` override applied at encode time. |
| `--grib-roundtrip` | Encode/decode every numpy intermediate through GRIB to reproduce the per-step quantisation of the legacy ksh chain. Honoured by products that support it (`sso`); ignored otherwise. |
| `-v`, `--verbose` | Count flag. Absent: silent (`WARNING`). `-v`: `INFO`. `-vv`: `DEBUG`. Streams to stdout. |
| `-f`, `--config PATH` | Optional YAML config file. Keys must match the product's Pydantic field names (snake_case). |
| `--set KEY=VAL` | Dot-separated override, e.g. `--set orography_grid=N2000`. |
| `-h`, `--help` | Show the field-specific help and exit. |

Product-specific flags are documented per-field via
`pproc-climate-fields <field> --help`. Every input path and every
output path is an explicit flag — the tool makes **no filename
assumptions**. See [Design principles](#design-principles).

## Design principles

The unified tool enforces two invariants that operators and product
authors both depend on.

### 1. Filenames stay in the wrapper

Every input path and every output path is an explicit CLI flag with a
default that is a bare filename in the current directory. Wrappers own
env-var → path mapping and any ECFS archival. The Python side does not
read env vars, does not construct paths from templates, and does not
know about `${XDATA_IFS}`, `${MIR_ERES_SET}`, or any other operational
variable.

Two multi-output shapes are worth calling out:

* **Fixed named outputs** (e.g. `sso`'s four outputs `stdgwd`, `slogwd`,
  `anggwd`, `isogwd`) get one `--<name>-out PATH` flag each.
* **N outputs differing only by a numeric suffix** (e.g. `sea-surface`'s
  twelve monthly SST outputs) use a single `--<prefix>-out-template
  PATH` flag with a `{month}` placeholder, e.g.
  `--sst-out-template ./sst_{month:02d}`. The framework validator
  rejects templates without the placeholder so silent overwrites are
  caught at parse time.

### 2. GRIB metadata stays in the product

`paramId`, `shortName`, `packingType`, `bitsPerValue`, `typeOfLevel`,
and local-definition keys are algorithm-intrinsic — the operator
should not need to know them, and the ksh wrapper no longer runs
`grib_set`. Each product applies its own metadata inside `generate()`
via
`pproc.common.io.encode_grib(values, template, metadata={...})`. If a
metadata key is wrong for a given output, that is a bug in the product
module, not something the operator patches in the wrapper.

## Wrapper contract

An `ifs-scripts/clim-pproc/generate_<field>.ksh` wrapper does exactly
three things:

1. Read the ecflow environment variables (`RESOL`, `GTYPE`, `XDATA`,
   `XDATA_IFS`, month indices, resolution ladders, hardcoded source-LSM
   paths, …).
2. Invoke `pproc-climate-fields <field>` with those values mapped onto
   the explicit CLI flags. No `grib_set`, no `mir-compute`, no
   `run_mir`.
3. Optionally archive the resulting output GRIB files to ECFS.

The Python side is unaware of steps 1 and 3; it sees only step 2.

## Configuration resolution order

Each product's Conflator app resolves configuration in this order
(later wins):

1. Field-level defaults declared on the product's Pydantic config
   (e.g. `land_mask_out: str = "./land_mask"`).
2. `--config PATH` YAML file(s), applied in argument order.
3. `--set KEY=VAL` overrides, applied in argument order.
4. Explicit CLI flags (e.g. `--target-grid N256`).

Unknown YAML keys raise a validation error rather than being silently
ignored.

## Example

```bash
pproc-climate-fields sso \
  --orography data/input/ifs/orog_5km \
  --land-mask data/input/ifs/land_mask \
  --target-grid N256 \
  --orography-grid N256 \
  --stdgwd-out ./out/stdgwd \
  --slogwd-out ./out/slogwd \
  --anggwd-out ./out/anggwd \
  --isogwd-out ./out/isogwd
```

This matches the legacy test run (`GTYPE_SET=O, ORES=80,
MIR_GTYPE_SET=N256, MIR_ERES_SET=N48`) and produces the four final
files. See [pproc-climate-fields sso](sso.md) for the full flag list
and pipeline details, and [Legacy script
mapping](../appendix/legacy-mapping.md) for the line-by-line ksh
correspondence.

## Related CLIs

[`pproc-formula`](formula.md) is a general-purpose shell tool for
evaluating arithmetic formulae over GRIB fields on disk. It wraps the
same `pproc.formula.evaluate_formula` engine that the
`pproc-climate-fields` product modules call internally, and is useful
for ad-hoc pipelines or wrapping formula-based operations from shell
scripts. It is not required for anything that
`pproc-climate-fields` already covers.
