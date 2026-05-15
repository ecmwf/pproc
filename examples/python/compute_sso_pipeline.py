"""End-to-end SSO computation via the pproc.climate.sso.pipeline API.

Use this pattern when:
- You want to embed SSO computation in a larger Python workflow.
- You want programmatic access to the four output GRIB buffers (without
  writing them to disk first).
- You need to pass through metadata, FDB targets, or other downstream targets.

For one-shot batch jobs, the ``pproc-sso`` CLI (see
``docs/climate/src/cli/sso.md``) is simpler.

The pipeline implements the three-grid operational model:

* ``source`` — the grid the input ``orography`` arrives on (auto-detected
  from the GRIB's ``gridName``). ``orography`` is treated as authoritative
  for ``orography_grid``: if its grid differs, Stage 1 raises
  ``ValueError`` rather than silently regridding. Route a wrong-grid file
  via ``alt_orography`` instead (see below).
* ``orography_grid`` — the high-resolution working grid where SSO
  statistics are computed. Operationally ``N2000`` (≈ 5 km); this example
  uses ``N256`` to keep the reference fixtures small.
* ``effective_resolution`` (eres) — the coarse aggregation grid derived
  from the model grid (Unit C).
* ``target_grid`` — the final IFS model grid the four outputs land on.

If ``orography`` does not exist on disk, the pipeline can fall back to
an ``alt_orography`` field (matching the legacy ksh's ``inFile_alt``
variable). The alternative is regridded to ``orography_grid`` and the
result is cached at the ``orography`` path so subsequent runs hit the
fast path. See ``docs/climate/src/cli/sso.md`` § *Cached vs alternative
orography input* for the full workflow.
"""

from pathlib import Path

from pproc.climate.sso.config import SSOConfig
from pproc.climate.sso.pipeline import compute_sso
from pproc.common.io import decode_grib

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    config = SSOConfig(
        orography=REPO_ROOT / "data" / "input" / "ifs" / "orog_5km",
        land_mask=REPO_ROOT / "data" / "input" / "ifs" / "land_mask",
        target_grid="N256",
        model_grid_type="O",
        model_resolution=80,
        orography_grid="N256",
    ).resolve()

    print(f"Running SSO pipeline on {config.orography.name}")
    print(f"  orography grid:        {config.orography_grid}")
    print(f"  target grid:           {config.target_grid}")
    print(f"  model:                 {config.model_grid_type}{config.model_resolution}")
    print(f"  effective resolution:  {config.effective_resolution}")

    result = compute_sso(config)

    for name in ("stdgwd", "slogwd", "anggwd", "isogwd"):
        values, meta = decode_grib(result[name])
        print(
            f"  {name}: shortName={meta['shortName']:<5s} "
            f"shape={values.shape} packing={meta['packingType']}"
        )


if __name__ == "__main__":
    main()
