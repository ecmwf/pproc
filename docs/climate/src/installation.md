# Installation

`pproc` is installed as an editable Python package from the workspace
root.

## Required dependencies

- `numpy`
- `eccodes` (Python bindings to the ECMWF eccodes C library)
- `earthkit-data`
- `mir-python` (Python bindings to mir; supplied as a pre-built wheel
  alongside the `mir` C++ build)
- `conflator` (Pydantic + argparse driver used by every
  `pproc-climate-fields` product)

The mir-python wheel that the pproc test suite was validated against is
shipped at:

```
mir/python/mir/dist/mir_python-1.28.2-cp314-cp314-macosx_26_0_arm64.whl
```

Replace the platform tag with the wheel that matches your interpreter
(`cp311-...`, `cp312-...`, `linux_x86_64`, …). The wheel is built by the
top-level `build-mir-python.sh` helper.

## Editable install

From the workspace root (`/path/to/climate-fields/pproc/`):

```bash
source .venv/bin/activate
uv pip install -e ./pproc
```

This puts the following console scripts on `PATH`:

* **`pproc-climate-fields`** — the unified climate-field generation
  tool; see [pproc-climate-fields](cli/climate-fields.md).
* `pproc-formula`, `pproc-gradient` — transitional wrappers still
  installed for callers on the pre-unification interface; see
  [pproc-formula](cli/formula.md) and [pproc-gradient](cli/gradient.md).
  They call the same underlying `pproc.formula.evaluate_formula` and
  `pproc.climate.mir_ops.gradient` library code that
  `pproc-climate-fields` uses, and will be removed once all operational
  callers have migrated. (The earlier standalone `pproc-sso` CLI has
  been removed; its behaviour is available as
  `pproc-climate-fields sso`.)
* The existing `pproc-*` console scripts unrelated to climate-fields
  (unchanged by this migration).

## Verifying the install

```bash
pproc-climate-fields --help
```

Should list the twenty-seven available field names. To verify a
specific field:

```bash
pproc-climate-fields sso --help
pproc-climate-fields lake-depth --help
```

Each command should print its argparse help banner without raising an
import error. If an import error mentions `mir.python` or `eccodes`,
re-check that the prebuilt wheels are on `PYTHONPATH` and that the
platform tag matches your interpreter.

The transitional CLIs should also come up cleanly:

```bash
pproc-formula --help
pproc-gradient --help
```
