"""Tests for ``pproc.climate.mir_ops`` — numpy-friendly mir.Job wrappers.

These tests drive the in-memory GRIB transport (no temp files) for the three
operations Unit D exposes: conservative interpolation (``grid-box-average``),
bilinear interpolation (``structured-bilinear``), and scalar gradient
(``mir.Job(nabla='scalar-gradient')``). Reference comparisons are done against
the existing intermediate files at the repo root that the legacy
``generate_subgrid_orography_sso.ksh`` pipeline produced.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import eccodes
import mir
import numpy as np
import pytest

from pproc.climate.mir_ops import (
    MirInvalidArgument,
    MirJobError,
    gradient,
    interpolate,
)
from pproc.common.io import decode_grib


# Repo root: .../pproc (the outer one — six levels up: this file is at
# pproc/pproc/tests/climate/test_mir_ops.py). ``parents[3]`` would land us in
# the inner ``pproc/`` package directory, so we go one further.
REPO_ROOT = Path(__file__).resolve().parents[3]


_REQUIRED_DATA: tuple[Path, ...] = (
    REPO_ROOT / "data" / "input" / "ifs" / "orog_5km",
    REPO_ROOT / "orog_egrid",
    REPO_ROOT / "orog_egrid_N2000",
    REPO_ROOT / "orog_egrid_diff",
)

_MISSING = tuple(p for p in _REQUIRED_DATA if not p.exists())

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=(
        "reference GRIB data not available at the workspace root; missing: "
        + ", ".join(str(p) for p in _MISSING)
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orog_5km_bytes() -> bytes:
    """N256 source orography (348 528 points, shortName ``z``)."""
    path = REPO_ROOT / "data" / "input" / "ifs" / "orog_5km"
    return path.read_bytes()


@pytest.fixture
def orog_egrid_bytes() -> bytes:
    """Reference N48 conservative-interpolated orography (13 280 points)."""
    return (REPO_ROOT / "orog_egrid").read_bytes()


@pytest.fixture
def orog_egrid_N2000_bytes() -> bytes:
    """Reference N256 bilinear-interpolated orography (348 528 points)."""
    return (REPO_ROOT / "orog_egrid_N2000").read_bytes()


@pytest.fixture
def orog_egrid_diff_bytes() -> bytes:
    """N256 difference field used as input for the gradient stage."""
    return (REPO_ROOT / "orog_egrid_diff").read_bytes()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interpolate_to_regular_lat_lon(grib_bytes: bytes, grid: str = "1/1") -> bytes:
    """Helper: interpolate a reduced-Gaussian field to a regular lat/lon grid.

    The N256 reduced-Gaussian grid does not place any points exactly on the
    poles (its first/last latitude row is at ±89.731°), so
    ``nabla_poles_missing_values`` cannot mark anything missing on it. To
    exercise that flag we first move onto a regular lat/lon grid that does
    include the ±90° rows.
    """
    inp = io.BytesIO(grib_bytes)
    out = io.BytesIO()
    mir.Job(grid=grid, interpolation="structured-bilinear").execute(inp, out)
    return out.getvalue()


# ---------------------------------------------------------------------------
# interpolate()
# ---------------------------------------------------------------------------


class TestInterpolate:
    def test_grid_box_average_to_N48_metadata(self, orog_5km_bytes):
        out = interpolate(orog_5km_bytes, grid="N48", method="grid-box-average")
        values, metadata = decode_grib(out)
        assert metadata["gridName"] == "N48"
        assert metadata["numberOfDataPoints"] == 13280
        assert values.shape == (13280,)

    def test_grid_box_average_matches_reference(self, orog_5km_bytes, orog_egrid_bytes):
        """D-K1 acceptance: bit-level reference match at ``rtol=1e-5``.

        If this ever fails at ``rtol=1e-5`` but passes at a looser tolerance
        (e.g. ``1e-3``), do **not** silently widen the tolerance — escalate
        via decision point D-K1. Pattern decides whether to widen or to use
        the ``--grib-roundtrip`` path.
        """
        out = interpolate(orog_5km_bytes, grid="N48", method="grid-box-average")
        out_values, _ = decode_grib(out)
        ref_values, _ = decode_grib(orog_egrid_bytes)
        np.testing.assert_allclose(out_values, ref_values, rtol=1e-5)

    def test_structured_bilinear_to_N256_metadata(self, orog_egrid_bytes):
        out = interpolate(orog_egrid_bytes, grid="N256", method="structured-bilinear")
        values, metadata = decode_grib(out)
        assert metadata["gridName"] == "N256"
        assert metadata["numberOfDataPoints"] == 348528
        assert values.shape == (348528,)

    def test_structured_bilinear_matches_reference(
        self, orog_egrid_bytes, orog_egrid_N2000_bytes
    ):
        out = interpolate(orog_egrid_bytes, grid="N256", method="structured-bilinear")
        out_values, _ = decode_grib(out)
        ref_values, _ = decode_grib(orog_egrid_N2000_bytes)
        np.testing.assert_allclose(out_values, ref_values, rtol=1e-5)

    def test_kwargs_underscore_to_hyphen(self, orog_5km_bytes):
        """Underscored kwargs go through the mir auto-conversion path.

        Smoke-test that an extra valid option (``vod2uv=False`` is irrelevant
        to scalar fields, but ``caching`` is a recognised mir option) does
        not blow up the call. We use ``caching="false"`` which mir accepts.
        """
        out = interpolate(
            orog_5km_bytes,
            grid="N48",
            method="grid-box-average",
            caching="false",
        )
        _, metadata = decode_grib(out)
        assert metadata["numberOfDataPoints"] == 13280

    def test_invalid_method_raises_typed_exception(self, orog_5km_bytes):
        with pytest.raises(MirInvalidArgument) as exc_info:
            interpolate(orog_5km_bytes, grid="N48", method="not-a-real-method")
        msg = str(exc_info.value)
        assert "not-a-real-method" in msg

    @pytest.mark.xfail(
        reason=(
            "mir silently passes through when the grid spec is unrecognised "
            "(no MethodFactory-style error is raised). This is a mir "
            "behaviour: unknown grid strings do not invalidate the job. The "
            "wrapper exposes typed exceptions for the cases mir does flag — "
            "see test_invalid_method_raises_typed_exception."
        ),
        strict=True,
    )
    def test_invalid_grid_raises_typed_exception(self, orog_5km_bytes):
        with pytest.raises((MirInvalidArgument, MirJobError)):
            interpolate(orog_5km_bytes, grid="not-a-grid", method="grid-box-average")

    def test_typed_exception_inherits_value_error(self, orog_5km_bytes):
        """Callers should be able to catch via the standard ``ValueError``."""
        with pytest.raises(ValueError):
            interpolate(orog_5km_bytes, grid="N48", method="not-a-real-method")

    def test_grib_bytes_must_be_bytes_like(self):
        with pytest.raises(TypeError):
            interpolate("not bytes", grid="N48", method="grid-box-average")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# gradient()
# ---------------------------------------------------------------------------


class TestGradient:
    def test_returns_two_messages_on_input_grid(self, orog_egrid_diff_bytes):
        gx, gy = gradient(orog_egrid_diff_bytes)

        # Each element must be a single-message GRIB buffer.
        gx_messages = list(eccodes.MemoryReader(gx))
        gy_messages = list(eccodes.MemoryReader(gy))
        assert len(gx_messages) == 1
        assert len(gy_messages) == 1

        gx_values, gx_meta = decode_grib(gx)
        gy_values, gy_meta = decode_grib(gy)

        assert gx_meta["gridName"] == "N256"
        assert gy_meta["gridName"] == "N256"
        assert gx_values.shape == (348528,)
        assert gy_values.shape == (348528,)

    def test_message_order_matches_reference(self, orog_egrid_diff_bytes):
        """Verify (gradx, grady) ordering against the SSO reference output."""
        gx, gy = gradient(orog_egrid_diff_bytes)
        ref_bytes = (REPO_ROOT / "orog_egrid_diff_grad").read_bytes()
        ref_messages = list(eccodes.MemoryReader(ref_bytes))
        assert len(ref_messages) == 2

        ref_gx = ref_messages[0].get_array("values")
        ref_gy = ref_messages[1].get_array("values")
        gx_values, _ = decode_grib(gx)
        gy_values, _ = decode_grib(gy)

        # The reference's "missing" cells are encoded with bitmapPresent=1
        # but the value happens to be 0 at the relevant points; for this
        # comparison we treat both arrays element-wise (ignoring NaN
        # positions to be safe).
        gx_mask = ~np.isnan(gx_values)
        np.testing.assert_allclose(gx_values[gx_mask], ref_gx[gx_mask], rtol=1e-5)
        gy_mask = ~np.isnan(gy_values)
        np.testing.assert_allclose(gy_values[gy_mask], ref_gy[gy_mask], rtol=1e-5)

    def test_poles_missing_values_default_true_on_regular_grid(
        self, orog_egrid_diff_bytes
    ):
        """``poles_missing_values=True`` flags the ±90° rows as missing.

        The ``orog_egrid_diff`` field is on a reduced-Gaussian N256 grid,
        whose first/last latitude rows sit at ±89.731° rather than ±90°, so
        the ``nabla-poles-missing-values`` flag has no effect. To exercise
        the flag we first interpolate to a regular 1°×1° lat/lon grid, which
        does include the ±90° rows.
        """
        regular_bytes = _interpolate_to_regular_lat_lon(
            orog_egrid_diff_bytes, grid="1/1"
        )
        gx, _ = gradient(regular_bytes)
        gx_values, _ = decode_grib(gx)
        # 360 + 360 points at lat=±90° should be missing.
        assert np.isnan(gx_values).any()
        n_nan = int(np.isnan(gx_values).sum())
        assert n_nan == 720, f"expected 720 missing pole points, got {n_nan}"

    def test_poles_missing_values_false_on_regular_grid(self, orog_egrid_diff_bytes):
        regular_bytes = _interpolate_to_regular_lat_lon(
            orog_egrid_diff_bytes, grid="1/1"
        )
        gx, _ = gradient(regular_bytes, poles_missing_values=False)
        gx_values, _ = decode_grib(gx)
        assert not np.isnan(gx_values).any()


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------


class TestNoTempFileLeak:
    def test_interpolate_leaves_no_temp_files(
        self, orog_5km_bytes, tmp_path, monkeypatch
    ):
        """No new files should appear in TMPDIR after a call."""
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        before = set(os.listdir(tmp_path))
        interpolate(orog_5km_bytes, grid="N48", method="grid-box-average")
        after = set(os.listdir(tmp_path))
        assert before == after

    def test_gradient_leaves_no_temp_files(
        self, orog_egrid_diff_bytes, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        before = set(os.listdir(tmp_path))
        gradient(orog_egrid_diff_bytes)
        after = set(os.listdir(tmp_path))
        assert before == after
