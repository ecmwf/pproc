# pproc.common.io codec extensions

`pproc.common.io` is the existing pproc I/O module. As part of the SSO
migration it gained four GRIB codec helpers that drive in-memory
transport between mir, the formula evaluator, and the SSO pipeline. All
four are public and are the supported entry points for callers that
need to encode or decode a GRIB byte buffer.

## Public functions

### `decode_grib(grib_bytes) -> tuple[np.ndarray, dict]`

```python
def decode_grib(grib_bytes: bytes) -> tuple[np.ndarray, dict]:
    ...
```

Decode a single GRIB message from `grib_bytes`. Returns a `float64`
values array (with GRIB missing values replaced by `np.nan`) and a
metadata dict covering the standard identification and packing keys.

**Raises** — `ValueError` if `grib_bytes` contains no messages;
`TypeError` if the input is not bytes-like.

### `decode_grib_with_metadata(grib_bytes) -> tuple[np.ndarray, GribMetadata]`

```python
def decode_grib_with_metadata(
    grib_bytes: bytes,
) -> tuple[np.ndarray, GribMetadata]:
    ...
```

Mirror of `decode_grib` that returns the canonical pproc metadata type
`GribMetadata` (subclass of `eccodes.Message`) instead of a dict. Use
this when downstream consumers need the eccodes handle (e.g.
`write_message`, `fdb_write_ufunc`, `target_factory`) — a dict is
JSON-friendly but lossy. See Pattern decision D-A in
`.weave/learnings/sso-migration.md`.

### `decode_multi_grib(grib_bytes, count) -> list[tuple[np.ndarray, dict]]`

```python
def decode_multi_grib(
    grib_bytes: bytes, count: int
) -> list[tuple[np.ndarray, dict]]:
    ...
```

Decode `count` consecutive GRIB messages from `grib_bytes`. The buffer
must contain at least `count` messages; if it contains fewer,
`ValueError` is raised so callers (in particular `mir-compute`-style
multi-message inputs that bundle fields with a shell `cat`) fail loudly
on malformed input.

**Raises** — `ValueError` if fewer than `count` messages are present or
if `count` is negative.

### `encode_grib(values, template, metadata=None) -> bytes`

```python
def encode_grib(
    values: np.ndarray,
    template: bytes | bytearray | memoryview | GribMetadata | eccodes.Message,
    metadata: dict | None = None,
) -> bytes:
    ...
```

Encode `values` as a GRIB message by cloning `template`. **Polymorphic
on template type** (Pattern decision D-A):

| `template` type | Path |
|-----------------|------|
| `bytes` / `bytearray` / `memoryview` | The wire bytes of a reference message; `encode_grib` parses out the first message and clones it. |
| `GribMetadata` / `eccodes.Message` | Used directly via its eccodes handle (no re-parse). |

When `metadata` is supplied, the overrides are applied via the existing
pproc convention (`construct_message`), which handles edition switches,
`MISSING` sentinels, array-valued keys, and the operational
`packingType` default for edition 2. When `metadata` is `None`, the
template is cloned via `template.copy()` — this avoids
`construct_message`'s edition-2 `packingType=grid_ccsds` default, which
would re-quantise a losslessly-packed (e.g. `grid_ieee`) template and
break bit-identical round-tripping.

**Raises** — `ValueError` if the template bytes contain no GRIB
message; `TypeError` if the template is neither bytes-like nor an
`eccodes.Message`.

## NaN ↔ missing-value bitmap

All four codec helpers participate in the NaN ↔ GRIB missing-value
convention used elsewhere in pproc:

- **Decoding**: after extracting the values array, `decode_grib` and
  `decode_multi_grib` call `missing_to_nan(message, values)` so that
  GRIB missing values surface as `np.nan` in the returned array. This
  is consistent with the rest of pproc (`fdb_read`, `iterate_xarray`,
  …).
- **Encoding**: before writing, `encode_grib` calls
  `nan_to_missing(message, data)` to translate `np.nan` entries into
  the message's `missingValue` and to flip `bitmapPresent=1` if any NaN
  is present. The cleaned data array is then written via
  `message.set_array("values", data)`.

The bitmap is therefore round-trip-stable: decoding a message with a
bitmap, then re-encoding the resulting NaN-bearing array, produces a
GRIB message whose bitmap matches the original.

## Relationship to existing helpers

| Existing | New | Difference |
|----------|-----|------------|
| `write_grib(target, template, data, metadata, missing=None)` | `encode_grib(values, template, metadata=None) -> bytes` | `write_grib` writes to a `Target`; `encode_grib` returns wire bytes. The two share the metadata-application convention via `construct_message`. |
| `read_grib_messages(messages, dims=())` | `decode_multi_grib(grib_bytes, count)` | `read_grib_messages` returns a `GRIBFields` dataclass keyed by stringified coords; `decode_multi_grib` returns a flat `list[(ndarray, dict)]`. |

There is no name collision with existing public symbols.
