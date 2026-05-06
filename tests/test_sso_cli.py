# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""End-to-end tests for the ``pproc-sso`` CLI.

Drives :func:`pproc.sso.main` directly (no subprocess) against the canonical
ksh test-run reference data (target=N256, model=O80 → eres=N48). The
tolerances applied here track the K2 sign-off documented in
``pproc/pproc/tests/climate/sso/test_pipeline.py`` — see that file for the
full discussion of the float32-vs-float64 arithmetic gap. The CLI is a thin
argparse wrapper around :func:`pproc.climate.sso.pipeline.compute_sso`, so
the tolerances here mirror the pipeline tolerances exactly; do **not** widen
them beyond the K2 posture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from pproc.common.io import decode_grib
from pproc.sso import main as sso_main


# tests/ -> pproc/ (inner package) -> pproc/ (repo subdir) — the data/ tree
# lives one level above the inner package.
REPO_ROOT = Path(__file__).resolve().parents[2]

_OROG_5KM = REPO_ROOT / "data" / "input" / "ifs" / "orog_5km"
_LAND_MASK = REPO_ROOT / "data" / "input" / "ifs" / "land_mask"
_REF_OUTPUT_DIR = REPO_ROOT / "data" / "output"

# K2 sign-off tolerances — DO NOT widen.
TOLERANCES = {
    "stdgwd": {"rtol": 1e-5, "atol": 1e-4},
    "slogwd": {"rtol": 1e-5, "atol": 1e-9},
    "isogwd": {"rtol": 1e-5, "atol": 1e-7},
    "anggwd": {"rtol": 1e-5, "atol": 1e-6},
}

OUTPUT_NAMES = ("stdgwd", "slogwd", "anggwd", "isogwd")

# 16 named intermediates emitted when --dump-intermediates is set.
INTERMEDIATE_NAMES = (
    "orog_egrid",
    "orog_egrid_N2000",
    "orog_egrid_diff",
    "orog_egrid_diff_grad",
    "orog_egrid_diff_gradx_sq",
    "orog_egrid_diff_grady_sq",
    "orog_egrid_diff_gradxy",
    "orog_eff_diff_sq",
    "orog_eff_diff_gradx_sq",
    "orog_eff_diff_grady_sq",
    "orog_eff_diff_gradxy",
    "orog_mgrid_diff_sq",
    "orog_mgrid_diff_gradx_sq",
    "orog_mgrid_diff_grady_sq",
    "orog_mgrid_diff_gradxy",
    "KLMLprime_lsm",
)


_REQUIRED_DATA: tuple[Path, ...] = (
    _OROG_5KM,
    _LAND_MASK,
    _REF_OUTPUT_DIR / "stdgwd",
    _REF_OUTPUT_DIR / "slogwd",
    _REF_OUTPUT_DIR / "anggwd",
    _REF_OUTPUT_DIR / "isogwd",
    *(REPO_ROOT / name for name in INTERMEDIATE_NAMES),
)

_MISSING = tuple(p for p in _REQUIRED_DATA if not p.exists())

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=(
        "reference GRIB data not available at the workspace root; missing: "
        + ", ".join(str(p) for p in _MISSING)
    ),
)


def _canonical_argv(out_dir: Path) -> list[str]:
    """Build the canonical ksh test-run argv for ``pproc-sso``."""
    return [
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
        "--output-grid",
        "N256",
        "--output-dir",
        str(out_dir),
    ]


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


class TestSSOCLISmoke:
    def test_produces_four_output_files(self, tmp_path):
        sso_main(_canonical_argv(tmp_path))
        for name in OUTPUT_NAMES:
            assert (tmp_path / name).is_file(), f"missing output: {name}"

    def test_no_intermediates_by_default(self, tmp_path):
        sso_main(_canonical_argv(tmp_path))
        # Only the four outputs should exist; none of the intermediates.
        for name in INTERMEDIATE_NAMES:
            assert not (tmp_path / name).exists(), (
                f"unexpected intermediate present without --dump-intermediates: {name}"
            )


# ---------------------------------------------------------------------------
# Reference match — default mode
# ---------------------------------------------------------------------------


class TestSSOCLIReferenceMatch:
    @pytest.mark.parametrize("name", OUTPUT_NAMES)
    def test_default_mode_matches_reference(self, tmp_path, name):
        sso_main(_canonical_argv(tmp_path))
        actual, _ = decode_grib((tmp_path / name).read_bytes())
        expected, _ = decode_grib((_REF_OUTPUT_DIR / name).read_bytes())
        np.testing.assert_allclose(
            actual,
            expected,
            equal_nan=True,
            **TOLERANCES[name],
        )


# ---------------------------------------------------------------------------
# Reference match — --grib-roundtrip
# ---------------------------------------------------------------------------


class TestSSOCLIGribRoundtrip:
    @pytest.mark.parametrize("name", OUTPUT_NAMES)
    def test_grib_roundtrip_matches_reference(self, tmp_path, name):
        sso_main(_canonical_argv(tmp_path) + ["--grib-roundtrip"])
        actual, _ = decode_grib((tmp_path / name).read_bytes())
        expected, _ = decode_grib((_REF_OUTPUT_DIR / name).read_bytes())
        np.testing.assert_allclose(
            actual,
            expected,
            equal_nan=True,
            **TOLERANCES[name],
        )


# ---------------------------------------------------------------------------
# --dump-intermediates
# ---------------------------------------------------------------------------


class TestSSOCLIDumpIntermediates:
    def test_all_intermediates_present(self, tmp_path):
        sso_main(_canonical_argv(tmp_path) + ["--dump-intermediates"])
        for name in INTERMEDIATE_NAMES:
            assert (tmp_path / name).is_file(), f"missing intermediate: {name}"

    def test_outputs_still_produced_with_intermediates(self, tmp_path):
        sso_main(_canonical_argv(tmp_path) + ["--dump-intermediates"])
        for name in OUTPUT_NAMES:
            assert (tmp_path / name).is_file(), f"missing output: {name}"


# ---------------------------------------------------------------------------
# --config YAML
# ---------------------------------------------------------------------------


class TestSSOCLIConfigFile:
    def _full_yaml(self, tmp_path: Path) -> dict:
        return {
            "orography": str(_OROG_5KM),
            "land_mask": str(_LAND_MASK),
            "target_grid": "N256",
            "model_grid_type": "O",
            "model_resolution": 80,
            "output_grid": "N256",
            "output_dir": str(tmp_path),
        }

    def test_yaml_config_loaded(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.safe_dump(self._full_yaml(tmp_path)))
        sso_main(["--config", str(cfg_file)])
        for name in OUTPUT_NAMES:
            assert (tmp_path / name).is_file(), name

    def test_cli_overrides_yaml(self, tmp_path):
        # YAML claims target_grid=N48 (bogus); CLI passes target-grid=N256.
        # The CLI value must win — verified by inspecting the gridName in
        # the output GRIB metadata.
        bad = self._full_yaml(tmp_path)
        bad["target_grid"] = "N48"
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.safe_dump(bad))
        sso_main(
            [
                "--config",
                str(cfg_file),
                "--target-grid",
                "N256",
            ]
        )
        _, meta = decode_grib((tmp_path / "stdgwd").read_bytes())
        assert meta.get("gridName") == "N256", (
            f"expected gridName=N256, got {meta.get('gridName')!r}"
        )

    def test_yaml_supplies_required_args(self, tmp_path):
        # Only --config on the CLI; YAML supplies all required fields.
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.safe_dump(self._full_yaml(tmp_path)))
        sso_main(["--config", str(cfg_file)])
        # Sanity: a single output is enough — full reference-match coverage
        # is provided by TestSSOCLIReferenceMatch above.
        assert (tmp_path / "stdgwd").is_file()


# ---------------------------------------------------------------------------
# Errors and --help
# ---------------------------------------------------------------------------


class TestSSOCLIErrors:
    def test_missing_required_args(self):
        with pytest.raises(SystemExit):
            sso_main([])

    def test_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            sso_main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for flag in (
            "--orography",
            "--land-mask",
            "--target-grid",
            "--source-orography",
            "--model-grid-type",
            "--model-resolution",
            "--effective-resolution",
            "--output-grid",
            "--output-dir",
            "--grib-roundtrip",
            "--dump-intermediates",
            "--config",
        ):
            assert flag in out, f"--help output missing flag: {flag}"
