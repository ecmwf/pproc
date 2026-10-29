# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``lake-depth`` pproc-climate-fields product.

Two classes:

* :class:`TestDepthFilterAlgorithm` — fast unit tests on the numerics of
  :mod:`pproc.climate.generate.products._lake_depth_filter`, run on
  synthetic tiny rasters. No real ``.dat`` file is opened.
* :class:`TestLakeDepthProduct` — one end-to-end test at the coarsest
  workable resolution (0.5°/N48), gated on the real 4.6 GB
  ``World_DEPTH.dat`` (+ status + template) being present.

Plus one dispatcher smoke test: ``pproc-climate-fields lake-depth --help``
exits 0 and lists all flags.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pproc.climate.generate.__main__ import main as generate_main
from pproc.climate.generate.products import _lake_depth_filter as ldf
from pproc.climate.generate.products.lake_depth import LakeDepthConfig, generate
from pproc.common.io import decode_grib


REPO_ROOT = Path(__file__).resolve().parents[4]
_WORLD_DEPTH = REPO_ROOT / "data" / "input" / "ifs" / "World_DEPTH.dat"
_WORLD_DEPTH_STATUS = REPO_ROOT / "data" / "input" / "ifs" / "World_DEPTHStatus.dat"
_TEMPLATE_05 = (
    REPO_ROOT
    / "data"
    / "input"
    / "ifs"
    / "grib_templates"
    / "RegularLatLonCellCentered_0.5_0.5"
)


# ---------------------------------------------------------------------------
# Unit tests — synthetic rasters only, no real data
# ---------------------------------------------------------------------------


def _write_raster(
    tmp_path: Path,
    depth: np.ndarray,  # (nlon, nlat) Fortran order
    status: np.ndarray,  # (nlon, nlat) Fortran order
) -> tuple[Path, Path]:
    """Write a synthetic depth+status raster pair.

    Data is stored so that
    ``np.fromfile(...).reshape((nlon, nlat), order='F')`` reconstructs it,
    matching the operational file layout consumed by
    :func:`read_depth_file` / :func:`read_status_file`.
    """
    dpath = tmp_path / "World_DEPTH.dat"
    spath = tmp_path / "World_DEPTHStatus.dat"
    # order='F' → column-major flatten produces (i0,j0),(i1,j0),...(i0,j1),...
    depth.astype("<f4").flatten(order="F").tofile(dpath)
    status.astype("<i1").flatten(order="F").tofile(spath)
    return dpath, spath


class TestDepthFilterAlgorithm:
    """Fast unit tests on _lake_depth_filter helper functions."""

    def test_mode_selection_gldbv3(self):
        """A 6×6 block with 4 GLDBv3 pixels at 5m and 32 GLDBv3 pixels at 22m.

        Total 36 pixels, all GLDBv3-flagged. Mode over the depth gradation
        buckets picks the 22 m bucket (DEPTH_DEFAULT_VALUE index 7).
        Because there are no GEBCO/geo-app/default pixels, the aggregated
        status is STATUS_GLDBv3 (30) and depth is 22 m.
        """
        # World_depth / World_status are (nlon, nlat) in Fortran order.
        depth = np.full((6, 6), 22.0, dtype=np.float32)
        depth.flat[:4] = 5.0  # 4 pixels at 5 m
        status = np.full((6, 6), 1, dtype=np.int8)  # 1 is in GLDBv3_STATUSES

        depth9, status9 = ldf.aggregate_depth_optimized(
            depth, status, new_res=6, nlat_new=1, nlon_new=1
        )
        assert depth9.shape == (1, 1)
        assert depth9[0, 0] == pytest.approx(22.0)
        assert int(status9[0, 0]) == ldf.STATUS_GLDBv3

    def test_ocean_plus_default_mixed(self):
        """A 4×4 block: 8 GEBCO pixels (depth=100m) + 8 default pixels.

        Expected status is STATUS_GEBCO_DEFAULT (31); expected depth is
        weighted mean: (100·8 + 10·8) / 16 = 55 m.
        """
        depth = np.zeros((4, 4), dtype=np.float32)
        status = np.zeros((4, 4), dtype=np.int8)
        # First half (8 pixels) → GEBCO ocean at 100 m
        depth.flat[:8] = 100.0
        status.flat[:8] = ldf.STATUS_GEBCO
        # Second half → default (status=0 or 71 both count as default)
        status.flat[8:] = ldf.STATUS_DEFAULT

        depth9, status9 = ldf.aggregate_depth_optimized(
            depth, status, new_res=4, nlat_new=1, nlon_new=1
        )
        assert int(status9[0, 0]) == ldf.STATUS_GEBCO_DEFAULT
        # Weighted mean of GEBCO(100 m, 8 px) and default(10 m, 8 px) = 55
        assert depth9[0, 0] == pytest.approx(55.0)

    def test_pure_default(self):
        """A block of all default-status pixels yields depth=10 m, status=STATUS_DEFAULT."""
        depth = np.zeros((4, 4), dtype=np.float32)
        status = np.full((4, 4), ldf.STATUS_DEFAULT, dtype=np.int8)

        depth9, status9 = ldf.aggregate_depth_optimized(
            depth, status, new_res=4, nlat_new=1, nlon_new=1
        )
        assert depth9[0, 0] == pytest.approx(10.0)
        assert int(status9[0, 0]) == ldf.STATUS_DEFAULT

    def test_mode_depth_vectorized_bucket_selection(self):
        """``get_mode_depth_vectorized`` picks the mode bucket via searchsorted.

        With inputs ``[3.0, 3.0, 3.0, 22.0, 100.0]``:

        * 3.0 m → gradation bucket (2, 4] → ``DEPTH_DEFAULT_VALUE[1]`` = 3 m
        * 22.0 m → bucket (20, 24] → index 7 (= 22 m)
        * 100.0 m → bucket (82, 118] → index 13 (= 100 m)

        Three 3.0 m samples dominate → mode is 3 m.
        """
        depths = np.array([3.0, 3.0, 3.0, 22.0, 100.0], dtype=np.float32)
        mode_val, count = ldf.get_mode_depth_vectorized(
            depths, None, ldf.DEPTH_GRADATION, ldf.DEPTH_DEFAULT_VALUE
        )
        assert mode_val == pytest.approx(3.0)
        assert count == 5

    def test_mode_depth_vectorized_picks_22m_bucket(self):
        """Sanity check on the 22 m bucket used by :meth:`test_mode_selection_gldbv3`.

        Verifies the mapping 22 m → ``DEPTH_DEFAULT_VALUE[7]`` = 22 m.
        """
        depths = np.array([22.0, 22.0, 22.0], dtype=np.float32)
        mode_val, _ = ldf.get_mode_depth_vectorized(
            depths, None, ldf.DEPTH_GRADATION, ldf.DEPTH_DEFAULT_VALUE
        )
        assert mode_val == pytest.approx(22.0)

    def test_longitude_shift_via_compute_regridded_depth(self, tmp_path):
        """End-to-end helper: build a 4×2 raster, aggregate 1:1, verify the
        second half of each row moves to the first half.

        Source layout (Fortran order, longitude fast axis):
            longitude j=0..3, latitude i=0..1
            depth[j, i]:  j=0  j=1  j=2  j=3
                    i=0:  1    2    3    4
                    i=1:  5    6    7    8

        ``compute_regridded_depth`` with new_res=1, nlat_new=2, nlon_new=4
        first aggregates 1:1 (same array), then re-lays out to a 1-D
        latitude-major array with a half-longitude shift
        (``lon_mid_pix = 4/2 = 2``). Column 0 gets the value from column
        (0+2)=2 in the aggregated array, column 1 from (1+2)=3, column 2
        from (2-2)=0, column 3 from (3-2)=1.

        Expected 1-D output (i outer, j inner):
            [d[2,0], d[3,0], d[0,0], d[1,0],
             d[2,1], d[3,1], d[0,1], d[1,1]]
        = [3, 4, 1, 2, 7, 8, 5, 6]
        """
        # All GLDBv3 status → aggregation returns the mode of a single
        # sample per block = DEPTH_DEFAULT_VALUE bucket. To avoid the
        # bucket rounding masking the shift, we use status=STATUS_GEBCO
        # (mean-aggregation, so a single pixel passes straight through).
        depth = np.array(
            [
                [1.0, 5.0],
                [2.0, 6.0],
                [3.0, 7.0],
                [4.0, 8.0],
            ],
            dtype=np.float32,
        )  # shape (nlon=4, nlat=2)
        status = np.full((4, 2), ldf.STATUS_GEBCO, dtype=np.int8)

        dpath, spath = _write_raster(tmp_path, depth, status)

        result = ldf.compute_regridded_depth(
            dpath, spath, new_res=1, nlat_new=2, nlon_new=4
        )
        expected = np.array([3.0, 4.0, 1.0, 2.0, 7.0, 8.0, 5.0, 6.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Dispatcher smoke — always safe to run (no real data needed)
# ---------------------------------------------------------------------------


class TestLakeDepthCLIHelp:
    """The dispatcher's ``--help`` output must list every product flag."""

    def test_help_lists_all_flags(self, capsys):
        with pytest.raises(SystemExit) as exc:
            generate_main(["lake-depth", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for flag in (
            "--world-depth",
            "--world-depth-status",
            "--depth-template",
            "--resol-km",
            "--nlat",
            "--nlon",
            "--target-grid",
            "--lakedl-out",
        ):
            assert flag in out, f"lake-depth --help missing: {flag}"


# ---------------------------------------------------------------------------
# End-to-end — gated on the real 4.6 GB reference data being present
# ---------------------------------------------------------------------------


_REAL_DATA_MISSING = tuple(
    p for p in (_WORLD_DEPTH, _WORLD_DEPTH_STATUS, _TEMPLATE_05) if not p.exists()
)


@pytest.mark.skipif(
    bool(_REAL_DATA_MISSING),
    reason=(
        "real World_DEPTH data not available; missing: "
        + ", ".join(str(p) for p in _REAL_DATA_MISSING)
    ),
)
class TestLakeDepthProduct:
    """Direct call to :func:`generate` against the real 0.5° template.

    Coarsest workable resolution: 720×360 output pixels each aggregating
    60×60 source pixels (= RESOLL=60 in the ksh resolution ladder), then
    bilinear onto N48 (13280 points). Total ≈ 259k output pixels × 3600
    source pixels each = ~930 M block operations, but each block is a
    tiny numpy view so the wall time is dominated by mask-generation and
    the Python loop.
    """

    def test_end_to_end_generates_valid_grib(self, tmp_path):
        config = LakeDepthConfig(
            world_depth=_WORLD_DEPTH,
            world_depth_status=_WORLD_DEPTH_STATUS,
            depth_template=_TEMPLATE_05,
            resol_km=60,
            nlat=360,
            nlon=720,
            target_grid="N48",
            lakedl_out=tmp_path / "lakedl",
        )

        try:
            result = generate(config)
        except ValueError as exc:
            # Local eccodes definitions may not include paramId 228007.
            # Convert to xfail rather than fail the suite.
            if "paramId" in str(exc) or "228007" in str(exc):
                pytest.xfail(
                    "eccodes definitions on this box do not include "
                    f"paramId 228007: {exc}"
                )
            raise

        # -- returns single logical key ------------------------------
        assert set(result.keys()) == {"lakedl"}
        payload = result["lakedl"]
        assert isinstance(payload, (bytes, bytearray))

        # -- decodes as one GRIB message on N48 ----------------------
        values, meta = decode_grib(payload)
        assert values.size == 13280, (
            f"expected N48 to have 13280 points, got {values.size}"
        )

        # -- values inside clamp range ------------------------------
        assert float(values.min()) >= 0.5 - 1e-6
        assert float(values.max()) <= 10000.0 + 1e-6

        # -- metadata reflects step 6 encode ------------------------
        assert meta["packingType"] == "grid_simple"
