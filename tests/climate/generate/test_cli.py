# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""End-to-end tests for the ``pproc-climate-fields`` CLI.

Covers the dispatcher (``pproc.climate.generate.__main__:main``) plus the
two exemplar products (``land-mask`` and ``sso``) via the same entry point
operational callers use. Reference-match tolerances for SSO track the K2
sign-off documented in
``pproc/pproc/tests/climate/generate/test_pipeline.py`` — see that file
for the full float32-vs-float64 discussion; do **not** widen them here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pproc.climate.generate.__main__ import main as generate_main
from pproc.common.io import decode_grib


# tests/ -> pproc/ (inner package) -> pproc/ (repo subdir) — the data/ tree
# lives one level above the inner package (parents[4] from this file).
REPO_ROOT = Path(__file__).resolve().parents[4]

_OROG_5KM = REPO_ROOT / "data" / "input" / "ifs" / "orog_5km"
_LAND_MASK = REPO_ROOT / "data" / "input" / "ifs" / "land_mask"
_REF_OUTPUT_DIR = REPO_ROOT / "data" / "output"
_LSM_INPUT = REPO_ROOT / "data" / "input" / "255_4" / "lsm"

SSO_OUTPUT_NAMES = ("stdgwd", "slogwd", "anggwd", "isogwd")

# K2 sign-off tolerances — DO NOT widen.
SSO_TOLERANCES = {
    "stdgwd": {"rtol": 1e-5, "atol": 1e-4},
    "slogwd": {"rtol": 1e-5, "atol": 1e-9},
    "isogwd": {"rtol": 1e-5, "atol": 1e-7},
    "anggwd": {"rtol": 1e-5, "atol": 1e-6},
}


# ---------------------------------------------------------------------------
# Dispatcher — no product-specific I/O needed, always safe to run
# ---------------------------------------------------------------------------


class TestDispatcher:
    """The thin argparse layer that picks the field and hands off to Conflator."""

    def test_no_args_lists_fields(self, capsys):
        rc = generate_main([])
        assert rc == 0
        out = capsys.readouterr().out
        # Both exemplars must appear with their one-line descriptions.
        assert "land-mask" in out
        assert "sso" in out
        assert "Available fields" in out

    def test_help_lists_fields(self, capsys):
        rc = generate_main(["--help"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "land-mask" in out
        assert "sso" in out

    def test_unknown_field_non_zero(self, capsys):
        rc = generate_main(["bogus-field"])
        assert rc != 0
        err = capsys.readouterr().err
        # Error message must name the offending field and list valid options.
        assert "bogus-field" in err
        assert "land-mask" in err
        assert "sso" in err

    def test_field_help_shows_field_flags(self, capsys):
        # ``pproc-climate-fields sso --help`` should surface the
        # SSO CLIArg flags (--orography, --stdgwd-out, etc.).
        with pytest.raises(SystemExit) as exc:
            generate_main(["sso", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for flag in (
            "--orography",
            "--land-mask",
            "--target-grid",
            "--orography-grid",
            "--stdgwd-out",
            "--slogwd-out",
            "--anggwd-out",
            "--isogwd-out",
            "--bits-per-value",
        ):
            assert flag in out, f"sso --help missing: {flag}"

    def test_land_mask_help_shows_field_flags(self, capsys):
        with pytest.raises(SystemExit) as exc:
            generate_main(["land-mask", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for flag in ("--land-cover-in", "--land-mask-out"):
            assert flag in out, f"land-mask --help missing: {flag}"


# ---------------------------------------------------------------------------
# land-mask product
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _LSM_INPUT.exists(),
    reason=f"land-cover input not available at {_LSM_INPUT}",
)
class TestLandMaskProduct:
    def test_threshold_and_repack(self, tmp_path):
        out_path = tmp_path / "land_mask"
        rc = generate_main(
            [
                "land-mask",
                "--land-cover-in",
                str(_LSM_INPUT),
                "--land-mask-out",
                str(out_path),
            ]
        )
        assert rc == 0
        assert out_path.is_file()

        # Content invariants: binary {0, 1}, grid_simple packing.
        values, meta = decode_grib(out_path.read_bytes())
        assert meta["packingType"] == "grid_simple"
        unique = set(np.unique(values).tolist())
        # Values are {0, 1} — the formula produces float representations
        # of those two integers; anything else is a decode/threshold bug.
        assert unique.issubset({0.0, 1.0}), (
            f"expected binary mask, got unique values {unique}"
        )

    def test_default_output_path(self, tmp_path, monkeypatch):
        # No ``--land-mask-out``: writes to ``./land_mask`` in the CWD.
        monkeypatch.chdir(tmp_path)
        rc = generate_main(
            [
                "land-mask",
                "--land-cover-in",
                str(_LSM_INPUT),
            ]
        )
        assert rc == 0
        assert (tmp_path / "land_mask").is_file()

    def test_bits_per_value_propagates(self, tmp_path):
        out_path = tmp_path / "land_mask"
        rc = generate_main(
            [
                "land-mask",
                "--land-cover-in",
                str(_LSM_INPUT),
                "--land-mask-out",
                str(out_path),
                "--bits-per-value",
                "16",
            ]
        )
        assert rc == 0
        _, meta = decode_grib(out_path.read_bytes())
        assert meta["bitsPerValue"] == 16


# ---------------------------------------------------------------------------
# SSO product — reference match through the dispatcher
# ---------------------------------------------------------------------------


_SSO_REQUIRED = (
    _OROG_5KM,
    _LAND_MASK,
    _REF_OUTPUT_DIR / "stdgwd",
    _REF_OUTPUT_DIR / "slogwd",
    _REF_OUTPUT_DIR / "anggwd",
    _REF_OUTPUT_DIR / "isogwd",
)
_SSO_MISSING = tuple(p for p in _SSO_REQUIRED if not p.exists())


def _sso_argv(tmp_path: Path) -> list[str]:
    """Canonical SSO argv against the ksh test-run reference."""
    return [
        "sso",
        "--orography",
        str(_OROG_5KM),
        "--land-mask",
        str(_LAND_MASK),
        "--target-grid",
        "N256",
        "--model-grid-type",
        "O",
        "--model-resolution",
        "80",
        "--orography-grid",
        "N256",
        "--stdgwd-out",
        str(tmp_path / "stdgwd"),
        "--slogwd-out",
        str(tmp_path / "slogwd"),
        "--anggwd-out",
        str(tmp_path / "anggwd"),
        "--isogwd-out",
        str(tmp_path / "isogwd"),
    ]


@pytest.mark.skipif(
    bool(_SSO_MISSING),
    reason=(
        "reference GRIB data not available; missing: "
        + ", ".join(str(p) for p in _SSO_MISSING)
    ),
)
class TestSSOProductReference:
    def test_produces_four_outputs(self, tmp_path):
        rc = generate_main(_sso_argv(tmp_path))
        assert rc == 0
        for name in SSO_OUTPUT_NAMES:
            assert (tmp_path / name).is_file(), f"missing output: {name}"

    @pytest.mark.parametrize("name", SSO_OUTPUT_NAMES)
    def test_matches_reference(self, tmp_path, name):
        rc = generate_main(_sso_argv(tmp_path))
        assert rc == 0
        actual, _ = decode_grib((tmp_path / name).read_bytes())
        expected, _ = decode_grib((_REF_OUTPUT_DIR / name).read_bytes())
        np.testing.assert_allclose(
            actual, expected, equal_nan=True, **SSO_TOLERANCES[name]
        )

    def test_output_metadata(self, tmp_path):
        rc = generate_main(_sso_argv(tmp_path))
        assert rc == 0
        expected_short = {
            "stdgwd": "sdor",
            "slogwd": "slor",
            "anggwd": "anor",
            "isogwd": "isor",
        }
        for name, short in expected_short.items():
            _, meta = decode_grib((tmp_path / name).read_bytes())
            assert meta["shortName"] == short, name
            assert meta["packingType"] == "grid_simple", name

    def test_bits_per_value_32(self, tmp_path):
        rc = generate_main(_sso_argv(tmp_path) + ["--bits-per-value", "32"])
        assert rc == 0
        for name in SSO_OUTPUT_NAMES:
            _, meta = decode_grib((tmp_path / name).read_bytes())
            assert meta["bitsPerValue"] == 32, name

    def test_absent_bits_per_value_uses_eccodes_default(self, tmp_path):
        rc = generate_main(_sso_argv(tmp_path))
        assert rc == 0
        for name in SSO_OUTPUT_NAMES:
            _, meta = decode_grib((tmp_path / name).read_bytes())
            # grid_simple default in eccodes is 24 bits.
            assert meta["bitsPerValue"] == 24, name
