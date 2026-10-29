# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Argparse-surface tests for the ``pproc-interpol`` CLI.

These tests target the CLI-widening work that exposes ``mir.Job`` options
already implemented in the underlying binding: the additional interpolation
methods (``structured-bilinear``, ``nearest-neighbour``, ``nearest-lsm``), the
``mode-integral`` value for ``--interpolation-statistics``, and the ``--lsm-*``
pass-through flags used by ``nearest-lsm``. Each of these is used by an IFS
climate ksh script being ported to pproc. The CLI is a thin wrapper —
``mir.Job`` implements the behaviour — so these tests are focused on argparse
acceptance/rejection plus one end-to-end run per new capability to confirm the
option name reaches mir as expected.

The tests drive :func:`pproc.interpol.main` directly (no subprocess) against
the workspace reference GRIBs. If those files are absent (e.g. running
outside the operational workspace) the whole module is skipped, following
the pattern used by ``test_sso_cli.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pproc.common.io import decode_grib
from pproc.interpol import main as interpol_main


# tests/ -> pproc/ (inner package) -> pproc/ (repo subdir) -> workspace root.
REPO_ROOT = Path(__file__).resolve().parents[2]

_OROG_EGRID = REPO_ROOT / "orog_egrid"  # N48 grid, ~53KB
_OROG_5KM = REPO_ROOT / "data" / "input" / "ifs" / "orog_5km"  # ~5km global
_LAND_MASK = REPO_ROOT / "data" / "input" / "ifs" / "land_mask"  # N256

_REQUIRED_DATA: tuple[Path, ...] = (_OROG_EGRID, _OROG_5KM, _LAND_MASK)
_MISSING = tuple(p for p in _REQUIRED_DATA if not p.exists())

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=(
        "reference GRIB data not available at the workspace root; missing: "
        + ", ".join(str(p) for p in _MISSING)
    ),
)


def _grid_name(path: Path) -> str:
    _, meta = decode_grib(path.read_bytes())
    return meta.get("gridName", "")


def _num_points(path: Path) -> int:
    _, meta = decode_grib(path.read_bytes())
    # eccodes exposes this as numberOfDataPoints on the message.
    return int(meta.get("numberOfDataPoints", 0))


# ---------------------------------------------------------------------------
# New interpolation methods (regex-widening)
# ---------------------------------------------------------------------------


class TestNewInterpolationMethods:
    """Methods added by widening the ``_interpolation`` regex."""

    def test_structured_bilinear_accepted(self, tmp_path):
        """N48 -> N256 with structured-bilinear (SSO Stage 3 pattern)."""
        out = tmp_path / "bilin.grib"
        interpol_main(
            [
                "--grid",
                "N256",
                "--interpolation",
                "structured-bilinear",
                str(_OROG_EGRID),
                str(out),
            ]
        )
        assert out.is_file(), "output GRIB not produced"
        assert _grid_name(out) == "N256"
        # N256 reduced Gaussian grid has 348528 points.
        assert _num_points(out) == 348528

    def test_nearest_neighbour_accepted(self, tmp_path):
        """Full-word alias ``nearest-neighbour`` (previously only ``nn``)."""
        out = tmp_path / "nn.grib"
        interpol_main(
            [
                "--grid",
                "N48",
                "--interpolation",
                "nearest-neighbour",
                str(_OROG_5KM),
                str(out),
            ]
        )
        assert out.is_file()
        assert _grid_name(out) == "N48"
        assert _num_points(out) == 13280

    def test_nn_short_alias_still_works(self, tmp_path):
        """Regression: the pre-existing ``nn`` alias must still be accepted."""
        out = tmp_path / "nn_short.grib"
        interpol_main(
            [
                "--grid",
                "N48",
                "--interpolation",
                "nn",
                str(_OROG_5KM),
                str(out),
            ]
        )
        assert out.is_file()
        assert _grid_name(out) == "N48"


# ---------------------------------------------------------------------------
# Regression on existing methods
# ---------------------------------------------------------------------------


class TestExistingMethodsRegression:
    """Guards against the regex widening breaking pre-existing methods."""

    def test_grid_box_average_still_works(self, tmp_path):
        out = tmp_path / "gba.grib"
        interpol_main(
            [
                "--grid",
                "N48",
                "--interpolation",
                "grid-box-average",
                str(_OROG_5KM),
                str(out),
            ]
        )
        assert out.is_file()
        assert _grid_name(out) == "N48"
        assert _num_points(out) == 13280


# ---------------------------------------------------------------------------
# --interpolation-statistics extension
# ---------------------------------------------------------------------------


class TestInterpolationStatistics:
    def test_mode_integral_accepted(self, tmp_path):
        """``mode-integral`` is used by generate_soil_type.ksh (operational)."""
        out = tmp_path / "modei.grib"
        interpol_main(
            [
                "--grid",
                "N48",
                "--interpolation",
                "grid-box-statistics",
                "--interpolation-statistics",
                "mode-integral",
                str(_OROG_5KM),
                str(out),
            ]
        )
        assert out.is_file()
        assert _grid_name(out) == "N48"

    def test_maximum_still_works(self, tmp_path):
        """Regression on the pre-existing ``maximum`` statistic."""
        out = tmp_path / "max.grib"
        interpol_main(
            [
                "--grid",
                "N48",
                "--interpolation",
                "grid-box-statistics",
                "--interpolation-statistics",
                "maximum",
                str(_OROG_5KM),
                str(out),
            ]
        )
        assert out.is_file()
        assert _grid_name(out) == "N48"


# ---------------------------------------------------------------------------
# LSM pass-through flags
# ---------------------------------------------------------------------------


class TestLSMFlags:
    """``--lsm-selection`` / ``--lsm-file-input`` / ``--lsm-file-output``
    flow into the ``mir.Job`` options dict as ``lsm-selection`` etc.
    """

    def test_nearest_lsm_with_file_selection(self, tmp_path):
        """End-to-end: nearest-lsm consuming --lsm-selection=file and both
        lsm-file paths. Uses the workspace ``land_mask`` for both sides
        (the operational scripts do the same when regridding onto the
        same target grid)."""
        out = tmp_path / "lsm.grib"
        interpol_main(
            [
                "--grid",
                "N48",
                "--interpolation",
                "nearest-lsm",
                "--lsm-selection",
                "file",
                "--lsm-file-input",
                str(_LAND_MASK),
                "--lsm-file-output",
                str(_LAND_MASK),
                str(_OROG_5KM),
                str(out),
            ]
        )
        assert out.is_file()
        assert _grid_name(out) == "N48"


# ---------------------------------------------------------------------------
# Argparse rejection paths
# ---------------------------------------------------------------------------


class TestArgparseRejection:
    def test_invalid_interpolation_rejected(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            interpol_main(
                [
                    "--grid",
                    "N48",
                    "--interpolation",
                    "not-a-method",
                    str(_OROG_5KM),
                    str(tmp_path / "x.grib"),
                ]
            )
        assert exc.value.code != 0

    def test_invalid_statistics_rejected(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            interpol_main(
                [
                    "--grid",
                    "N48",
                    "--interpolation",
                    "grid-box-statistics",
                    "--interpolation-statistics",
                    "not-a-stat",
                    str(_OROG_5KM),
                    str(tmp_path / "x.grib"),
                ]
            )
        assert exc.value.code != 0

    def test_invalid_intermediate_interpolation_rejected(self, tmp_path):
        """``--intermediate-interpolation`` shares the ``_interpolation``
        regex — the widening must apply to it too, and rejection must
        also apply."""
        with pytest.raises(SystemExit) as exc:
            interpol_main(
                [
                    "--grid",
                    "N48",
                    "--intermediate-interpolation",
                    "not-a-method",
                    str(_OROG_5KM),
                    str(tmp_path / "x.grib"),
                ]
            )
        assert exc.value.code != 0


# ---------------------------------------------------------------------------
# --help surface & debug-print removal
# ---------------------------------------------------------------------------


class TestHelpAndCleanStdout:
    def test_help_lists_new_options(self, capsys):
        with pytest.raises(SystemExit) as exc:
            interpol_main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        # argparse wraps long option/regex strings across lines, splitting
        # tokens like ``structured-bilinear`` as ``structured-\n<spaces>bilinear``.
        # Collapse those wrap breaks before substring-matching. This is only
        # safe for the substring checks below (hyphenated identifiers); the
        # raw ``out`` is inspected verbatim elsewhere.
        flat = re.sub(r"-\n\s+", "-", out)
        # New flags surfaced.
        for flag in ("--lsm-selection", "--lsm-file-input", "--lsm-file-output"):
            assert flag in flat, f"--help missing flag: {flag}"
        # New interpolation methods surfaced in the help text.
        for method in (
            "structured-bilinear",
            "nearest-neighbour",
            "nearest-lsm",
        ):
            assert method in flat, f"--help missing interpolation method: {method}"
        # New statistics vocabulary surfaced.
        assert "mode-integral" in flat, "--help missing statistic: mode-integral"
