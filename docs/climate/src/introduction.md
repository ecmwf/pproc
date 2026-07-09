# Introduction

This documentation covers the SSO migration from the developer-only
`mir` + `mir-compute` CLIs to the operational `pproc` toolkit. Three new
console scripts are introduced — `pproc-sso` (the monolithic ten-stage
sub-grid orography pipeline), `pproc-gradient` (a thin wrapper around
mir's nabla operator), and `pproc-formula` (a numpy-backed replacement
for `mir-compute` with a hand-written formula parser, no `eval()`) —
together with a new `pproc.formula` package and a `pproc.climate.*`
library (`mir_ops`, `sso/{pipeline, config, effective_resolution}`)
plus GRIB encode/decode
helpers extending `pproc.common.io`.

The workload reproduces the legacy IFS shell script
[`generate_subgrid_orography_sso.ksh`](https://github.com/ecmwf-ifs/) (see
the source at the workspace root). A line-by-line mapping from each ksh
stage to the equivalent Python function or CLI invocation is in
[Legacy script mapping](appendix/legacy-mapping.md).

## High-level data flow

```mermaid
flowchart LR
    src[source orography] --> mi1[mir interpolate]
    mi1 --> na1[numpy arithmetic]
    na1 --> mg[mir gradient]
    mg --> na2[numpy arithmetic]
    na2 --> ma[mir aggregate]
    ma --> out[final SSO fields]
```

The pipeline is sequential. Intermediate data is held as in-memory GRIB
byte buffers between mir invocations, and as `numpy.float64` arrays for
the arithmetic stages. An opt-in `--grib-roundtrip` flag re-encodes and
decodes after every numpy step so the per-step GRIB quantisation of the
original ksh script can be reproduced for debugging; `--dump-intermediates`
writes the sixteen named intermediate GRIB files to disk.
