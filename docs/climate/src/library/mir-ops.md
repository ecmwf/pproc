# pproc.climate.mir_ops

Numpy-friendly wrappers around `mir.Job`. The module exposes thin,
in-memory wrappers around two mir operations the SSO pipeline relies
on. Transport between mir and the caller is GRIB bytes via
`io.BytesIO` — no temporary files are created or left behind.

> **Quarantine boundary.** This is the recommended way to use `mir.Job`
> from Python within `pproc`. Do **not** import `mir.python` directly
> from pipeline or CLI code; route through `pproc.climate.mir_ops`. The
> `mir/python/` tree is a hard quarantine boundary in the SSO migration
> (no edits, no direct imports).

## Public API

### `interpolate(grib_bytes, *, grid, method, **kwargs) -> bytes`

```python
def interpolate(
    grib_bytes: bytes,
    *,
    grid: str,
    method: str,
    **kwargs: Any,
) -> bytes:
    ...
```

Interpolate a GRIB field via `mir.Job`, returning GRIB bytes.

**Parameters**

- `grib_bytes` — input GRIB message(s) as bytes.
- `grid` — target grid spec, e.g. `"N48"`, `"N256"`, `"O1280"`,
  `"1.0/1.0"`.
- `method` — interpolation method. The full list lives in mir; common
  values are `"grid-box-average"`, `"structured-bilinear"`,
  `"nearest-neighbour"`, `"nearest-lsm"`.
- `**kwargs` — additional `mir.Job` options. Underscores in keys
  convert to hyphens (e.g. `lsm_selection` → `lsm-selection`); Python
  bools are coerced to the lowercase strings the mir bindings accept.

**Returns** — output GRIB message bytes on the target grid.

**Raises**

- `MirInvalidArgument` — if `method` or `grid` is unrecognised by mir.
- `MirJobError` — for any other failure surfaced by mir.
- `TypeError` — if `grib_bytes` is not bytes-like.

**Example**

```python
from pproc.climate.mir_ops import interpolate

with open("orog_5km", "rb") as f:
    src = f.read()
egrid = interpolate(src, grid="N48", method="grid-box-average")
```

### `gradient(grib_bytes, *, poles_missing_values=True) -> tuple[bytes, bytes]`

```python
def gradient(
    grib_bytes: bytes,
    *,
    poles_missing_values: bool = True,
) -> tuple[bytes, bytes]:
    ...
```

Compute the scalar gradient (∂f/∂lat, ∂f/∂lon) via mir's nabla
operator.

**Parameters**

- `grib_bytes` — input GRIB message bytes (single scalar field).
- `poles_missing_values` — if `True` (default), values at lat=±90° in
  the output gradient are flagged as missing — equivalent to the mir
  option `nabla-poles-missing-values=true` used by the legacy SSO ksh
  pipeline.

**Returns** — `(gradx_bytes, grady_bytes)`: two single-message GRIB
buffers, in the order produced by mir (∂f/∂lat first, then ∂f/∂lon),
matching the byte layout of the legacy `orog_egrid_diff_grad`
intermediate. Each buffer is byte-exact with mir's output (no
decode/re-encode round-trip).

**Raises**

- `MirJobError` — on any mir execution failure.
- `TypeError` — if `grib_bytes` is not bytes-like.
- `ValueError` — if mir does not produce exactly two output messages.

**Example**

```python
from pproc.climate.mir_ops import gradient

with open("orog_egrid_diff", "rb") as f:
    diff = f.read()
gradx, grady = gradient(diff)
```

## Typed exceptions

| Class | Base | Raised when |
|-------|------|-------------|
| `MirJobError` | `RuntimeError` | A `mir.Job` execution fails for any reason. Wraps the original `RuntimeError` (chain via `__cause__`). |
| `MirInvalidArgument` | `MirJobError`, `ValueError` | The mir error message looks like an unknown method, grid, nabla, or interpolation argument. |

`MirInvalidArgument` subclasses `ValueError` so callers that already
filter on the standard `ValueError` hierarchy continue to work.

## Internal helpers

The module also defines helpers that are not part of the public API but
are documented for maintainers: `_ensure_bytes_like`, `_coerce_mir_value`
(`bool` → `"true"`/`"false"`), `_build_job`, `_execute_job`, and
`_split_messages` (uses `eccodes.MemoryReader.get_buffer()` to split
multi-message mir output without a decode/re-encode round-trip — this
is what makes `gradient()` byte-exact).
