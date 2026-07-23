# Introduction

This documentation covers the migration of the IFS **climate-fields**
generation pipeline from the developer-only
[`mir`](https://github.com/ecmwf/mir) +
[`mir-compute`](https://github.com/ecmwf/mir) CLIs to the operational
`pproc` toolkit. The pipeline used to run as a set of ~30
[`ifs-scripts/clim/generate_*.ksh`](https://github.com/ecmwf-ifs/) shell
scripts, each shelling out to `mir`/`mir-compute`/`grib_set` on temp
files. It now runs as a single Python entry point,
[`pproc-climate-fields`](cli/climate-fields.md), that dispatches to
one of twenty-seven in-tree Python products.

## The unified tool

```
pproc-climate-fields <field> [<field-flags>]
```

`<field>` selects one of twenty-seven products; each product lives at
`pproc.climate.generate.products.<field>` and exposes `FIELD_NAME`,
`DESCRIPTION`, a Pydantic `CONFIG` class, and a `generate(config) ->
dict[str, bytes]` function. The full catalogue is in
[pproc-climate-fields § Product catalogue](cli/climate-fields.md#product-catalogue).

Under the surface the tool is an argparse dispatcher on `argv[1]` that
hands the remaining args to a per-product Conflator app for
`CLIArg`+YAML parsing. The dispatcher is the same pattern
`pproc.clustereps` uses; it has no per-field logic of its own. See
[pproc-climate-fields](cli/climate-fields.md).

## What moved where

Two invariants govern the split between the ksh wrappers under
`ifs-scripts/clim-pproc/` and the Python products:

1. **Filenames stay in the wrapper.** Every input and output path is
   an explicit CLI flag. The wrapper maps ecflow environment variables
   (`$RESOL`, `$XDATA`, `$XDATA_IFS`, …) onto those flags and, when
   required, archives outputs to ECFS. The Python side reads no env
   vars and knows about no operational paths.
2. **GRIB metadata stays in the product.** `paramId`, `shortName`,
   `packingType`, `bitsPerValue`, `typeOfLevel`, and local-definition
   keys are algorithm-intrinsic — the product applies them inside
   `generate()` via `pproc.common.io.encode_grib(metadata=...)`. The
   ksh wrapper no longer runs `grib_set`.

The library layer underneath the products consists of three pproc
modules:

* [`pproc.climate.mir_ops`](library/mir-ops.md) — numpy-friendly
  wrappers around `mir.Job` for interpolation and gradient
  computation; transports GRIB bytes through `io.BytesIO`.
* [`pproc.formula`](https://github.com/ecmwf/pproc) — the arithmetic
  formula parser (recursive-descent, no `eval()`); replaces
  `mir-compute`'s expression evaluation.
* [`pproc.common.io`](library/io-grib.md) — extended with
  `encode_grib` / `decode_grib` / `decode_multi_grib` for in-memory
  GRIB codec round-trips.

## Illustrative data flow

Most products are one-step (interpolate + threshold + write) or
twelve-step (monthly loop over the same one-step); their flow diagrams
are trivial. The most complex product is `sso`, whose ten-stage
pipeline is the reason the framework has a `--grib-roundtrip` flag and
a `--dump-intermediates` mode:

```mermaid
flowchart LR
    src[source orography] --> mi1[mir interpolate]
    mi1 --> na1[numpy arithmetic]
    na1 --> mg[mir gradient]
    mg --> na2[numpy arithmetic]
    na2 --> ma[mir aggregate]
    ma --> out[stdgwd/slogwd/anggwd/isogwd]
```

The pipeline is sequential. Intermediate data is held as in-memory
GRIB byte buffers between mir invocations, and as `numpy.float64`
arrays for the arithmetic stages. See [pproc-climate-fields
sso](cli/sso.md) for the full ten-stage decomposition, or [Legacy
script mapping](appendix/legacy-mapping.md) for the line-by-line
correspondence with `generate_subgrid_orography_sso.ksh`.

## Not migrated

Two legacy ksh scripts are outside the `pproc-climate-fields` model
because they call non-mir external tools:

* `generate_spectral_orography.ksh` — MPI-parallel Fortran
  `gptosp.exe` via `run_parallel`.
* `generate_vegetation.ksh` — Python + on-the-fly Cython build
  (`agg_esacci_lc.py`, `agg_cgls.py`, `disagg_cgls.py`).

Both remain callable as their original ksh scripts. Bringing them
under `pproc-climate-fields` is possible in principle (one product
module per script) but is deferred pending a design decision on how to
handle the MPI-parallel and Cython dependencies.

## Reference-match posture

The unified tool aims for bit-identical output against the legacy ksh
reference data. Two products currently exhibit sub-packing-precision
drift traceable to a float32-vs-float64 arithmetic path (`albedo` and
`orography-variance`, in addition to `sso`); the tolerance policy is
in [Tolerance policy (D-F1)](appendix/tolerance-policy.md).
