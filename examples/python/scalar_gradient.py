"""Direct use of pproc.climate.mir_ops.gradient on GRIB bytes.

Use this pattern when:
- You already have GRIB bytes in hand (e.g. from a memory buffer or FDB read).
- You want spatial derivatives (df/dlat, df/dlon) of a scalar field.
- You want the byte-exact wire output (gradient preserves mir's bytes verbatim).

For one-shot file-based use, prefer the ``pproc-gradient`` CLI.
"""

from pathlib import Path

from pproc.climate.mir_ops import gradient
from pproc.common.io import decode_grib

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    inp = REPO_ROOT / "orog_egrid_diff"  # the SSO pipeline's diff field
    print(f"Reading {inp.name}")
    grib_bytes = inp.read_bytes()

    # Compute gradient. poles_missing_values=True flags lat=+/-90 deg as missing.
    gradx_bytes, grady_bytes = gradient(grib_bytes, poles_missing_values=True)

    gx_values, gx_meta = decode_grib(gradx_bytes)
    gy_values, gy_meta = decode_grib(grady_bytes)

    print(f"  df/dlat: shape={gx_values.shape} grid={gx_meta['gridName']}")
    print(f"  df/dlon: shape={gy_values.shape} grid={gy_meta['gridName']}")


if __name__ == "__main__":
    main()
