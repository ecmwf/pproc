# Installation

`pproc` is installed as an editable Python package from the workspace
root.

## Required dependencies

- `numpy`
- `eccodes` (Python bindings to the ECMWF eccodes C library)
- `earthkit-data`
- `mir-python` (Python bindings to mir; supplied as a pre-built wheel
  alongside the `mir` C++ build)

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

This puts `pproc-sso`, `pproc-gradient`, `pproc-field-calc`, and the
existing `pproc-*` console scripts on `PATH`.

## Verifying the install

```bash
pproc-sso --help
pproc-gradient --help
pproc-field-calc --help
```

Each command should print its argparse help banner without raising an
import error. If an import error mentions `mir.python` or `eccodes`,
re-check that the prebuilt wheels are on `PYTHONPATH` and that the
platform tag matches your interpreter.
