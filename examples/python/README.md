# Python examples for pproc climate field generation

These examples demonstrate the three main entry points into the SSO migration
work:

- `compute_sso_pipeline.py` — end-to-end SSO computation via the
  `pproc.climate.sso.pipeline.compute_sso` API.
- `evaluate_formula.py` — direct use of the formula DSL on numpy arrays
  (no GRIB I/O).
- `scalar_gradient.py` — direct use of `mir_ops.gradient` on GRIB bytes.

## Running the examples

```bash
source .venv/bin/activate
python pproc/examples/python/compute_sso_pipeline.py
python pproc/examples/python/evaluate_formula.py
python pproc/examples/python/scalar_gradient.py
```

The first and third examples require the test data at the repo root
(`data/input/ifs/orog_5km`, `orog_egrid_diff`, etc.) — these are the same
files exercised by the test suite.

## Where to go next

- For CLI-based use: see `docs/climate/src/cli/{field-calc,gradient,sso}.md`.
- For library API: see `docs/climate/src/library/{mir-ops,io-grib}.md`.
- For the legacy ksh-script mapping: see `docs/climate/src/appendix/legacy-mapping.md`.
