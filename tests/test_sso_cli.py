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
    "orog_egrid_N256",
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
        "--orography-grid",
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
            "orography_grid": "N256",
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


class TestSSOCLIBitsPerValue:
    """``--bits-per-value N`` propagates through to GRIB ``bitsPerValue``
    on the four outputs. The flag is rejected if ``N`` is not a positive
    integer (pydantic ``gt=0`` constraint)."""

    OUTPUT_NAMES = ("stdgwd", "slogwd", "anggwd", "isogwd")

    def test_bits_per_value_32(self, tmp_path):
        sso_main(_canonical_argv(tmp_path) + ["--bits-per-value", "32"])
        for name in self.OUTPUT_NAMES:
            _, meta = decode_grib((tmp_path / name).read_bytes())
            assert meta["bitsPerValue"] == 32, name

    def test_bits_per_value_16(self, tmp_path):
        sso_main(_canonical_argv(tmp_path) + ["--bits-per-value", "16"])
        for name in self.OUTPUT_NAMES:
            _, meta = decode_grib((tmp_path / name).read_bytes())
            assert meta["bitsPerValue"] == 16, name

    def test_absent_flag_uses_eccodes_default(self, tmp_path):
        # No flag: pproc does not write bitsPerValue, so eccodes picks
        # the grid_simple default (24).
        sso_main(_canonical_argv(tmp_path))
        for name in self.OUTPUT_NAMES:
            _, meta = decode_grib((tmp_path / name).read_bytes())
            assert meta["bitsPerValue"] == 24, name

    def test_zero_rejected(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            sso_main(_canonical_argv(tmp_path) + ["--bits-per-value", "0"])
        assert exc.value.code != 0
        # pydantic ValidationError carries the "greater than 0" constraint
        # text; we don't pin the exact phrasing but require some signal
        # that this was a constraint violation rather than an unrelated
        # crash.
        captured = capsys.readouterr()
        combined = (captured.out + captured.err + str(exc.value)).lower()
        assert (
            "greater than" in combined
            or "bits_per_value" in combined
            or "bitspervalue" in combined
            or "validation" in combined
        ), f"expected a constraint-related error, got: {combined!r}"


class TestSSOCLIErrors:
    def test_missing_required_args(self):
        with pytest.raises(SystemExit):
            sso_main([])

    def test_missing_orography_grid_clean_exit(self, tmp_path, capsys):
        """``--orography-grid`` is required; omitting it must exit non-zero
        with a clear error message referencing the missing flag."""
        argv = [
            "--orography",
            str(_OROG_5KM),
            "--land-mask",
            str(_LAND_MASK),
            "--target-grid",
            "N256",
            "--output-dir",
            str(tmp_path),
        ]
        with pytest.raises(SystemExit) as exc:
            sso_main(argv)
        assert exc.value.code != 0
        captured = capsys.readouterr()
        combined = (captured.err + captured.out + str(exc.value)).lower()
        assert "orography-grid" in combined or "orography_grid" in combined, (
            f"expected an error mentioning --orography-grid, got: {combined!r}"
        )

    def test_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            sso_main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for flag in (
            "--orography",
            "--alt-orography",
            "--land-mask",
            "--target-grid",
            "--orography-grid",
            "--model-grid-type",
            "--model-resolution",
            "--effective-resolution",
            "--output-dir",
            "--grib-roundtrip",
            "--dump-intermediates",
            "--bits-per-value",
            "--config",
        ):
            assert flag in out, f"--help output missing flag: {flag}"


# ---------------------------------------------------------------------------
# Stage 1 grid mismatch (source ≠ orography_grid → clean ValueError exit)
# ---------------------------------------------------------------------------


_RAW_O256_OROG = REPO_ROOT / "data" / "input" / "255_4" / "orog"


@pytest.mark.skipif(
    not _RAW_O256_OROG.exists(),
    reason=f"raw O256 orography not available at {_RAW_O256_OROG}",
)
class TestSSOCLIAltOrography:
    """`--alt-orography` is the fallback orography input used when
    `--orography` is missing on disk. Verifies the CLI threads the flag
    through, that YAML config supports `alt_orography`, and that the
    error path on a fully-missing input is a clean non-zero exit (no
    traceback)."""

    def test_alt_orography_flag_used_when_orography_missing(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached = cache_dir / "orog_5km"
        out_dir = tmp_path / "out"
        argv = [
            "--orography",
            str(cached),
            "--alt-orography",
            str(_RAW_O256_OROG),
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
            "--output-dir",
            str(out_dir),
        ]
        sso_main(argv)
        for name in OUTPUT_NAMES:
            assert (out_dir / name).is_file(), f"missing output: {name}"
        # Cache writeback happened: the previously-missing orography
        # path now exists on disk.
        assert cached.is_file(), "expected cache file to be written"

    def test_alt_orography_in_yaml_config(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached = cache_dir / "orog_5km"
        out_dir = tmp_path / "out"
        yaml_doc = {
            "orography": str(cached),
            "alt_orography": str(_RAW_O256_OROG),
            "land_mask": str(_LAND_MASK),
            "target_grid": "N256",
            "model_grid_type": "O",
            "model_resolution": 80,
            "orography_grid": "N256",
            "output_dir": str(out_dir),
        }
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.safe_dump(yaml_doc))
        sso_main(["--config", str(cfg_file)])
        for name in OUTPUT_NAMES:
            assert (out_dir / name).is_file(), f"missing output: {name}"
        assert cached.is_file()

    def test_orography_missing_no_alt_clean_exit(self, tmp_path, capsys):
        cached = tmp_path / "nonexistent" / "orog_5km"
        out_dir = tmp_path / "out"
        argv = [
            "--orography",
            str(cached),
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
            "--output-dir",
            str(out_dir),
        ]
        with pytest.raises(SystemExit) as exc:
            sso_main(argv)
        assert exc.value.code != 0
        combined = str(exc.value)
        # The clean error message starts with the pproc-sso prefix and
        # mentions the missing file plus the helpful hint pointing
        # operators at --alt-orography.
        assert combined.startswith(
            f"pproc-sso: error: orography file '{cached}' does not exist;"
        ), f"unexpected error message: {combined!r}"
        assert "--alt-orography" in combined


@pytest.mark.skipif(
    not _RAW_O256_OROG.exists(),
    reason=f"raw O256 orography not available at {_RAW_O256_OROG}",
)
class TestSSOCLIGridMismatch:
    """When ``--orography`` is on a different grid from ``--orography-grid``,
    Stage 1 raises ``ValueError`` (configuration error). The CLI wrapper
    converts that to a clean non-zero ``SystemExit`` carrying the
    ``pproc-sso: error: ...`` prefix; no Python traceback escapes. The
    operator hint must point at ``--alt-orography``."""

    def test_grid_mismatch_clean_exit(self, tmp_path):
        argv = [
            "--orography",
            str(_RAW_O256_OROG),
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
            "--output-dir",
            str(tmp_path),
        ]
        with pytest.raises(SystemExit) as exc:
            sso_main(argv)
        # Non-zero exit (string codes are truthy => non-zero shell exit).
        assert exc.value.code != 0
        # The SystemExit was raised with the message as its argument; the
        # original ValueError is chained via ``from exc`` so any
        # Python-level traceback would surface that, not bypass the
        # wrapper. We check the user-facing message text directly.
        msg = str(exc.value)
        assert msg.startswith(
            f"pproc-sso: error: orography file '{_RAW_O256_OROG}' is on grid 'O256' "
            f"but --orography-grid is 'N256';"
        ), f"unexpected error message: {msg!r}"
        assert "supply an orography on 'N256'" in msg
        assert "move this file to --alt-orography to have it regridded" in msg

    def test_grid_mismatch_does_not_write_outputs(self, tmp_path):
        """A failed Stage 1 must not leave partial outputs behind."""
        out_dir = tmp_path / "out"
        argv = [
            "--orography",
            str(_RAW_O256_OROG),
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
            "--output-dir",
            str(out_dir),
        ]
        with pytest.raises(SystemExit):
            sso_main(argv)
        # The output directory must not have been populated (the CLI
        # writes outputs only after ``compute_sso`` returns).
        for name in OUTPUT_NAMES:
            assert not (out_dir / name).exists(), f"unexpected output: {name}"


# ---------------------------------------------------------------------------
# --verbose / -v
# ---------------------------------------------------------------------------


class TestSSOCLIVerbose:
    """Smoke tests for the ``-v`` / ``--verbose`` count flag.

    Asserts only presence/absence and a non-empty count — the precise log
    message text is intentionally not pinned so future log-line additions
    don't churn these tests. See ``pproc.climate._logging`` for the
    level-mapping contract that these tests exercise via ``sso_main``.
    """

    @pytest.fixture(autouse=True)
    def _reset_root_logger(self):
        import logging as _logging

        from pproc.climate._logging import _HANDLER_TAG

        root = _logging.getLogger()
        saved = list(root.handlers)
        level = root.level
        root.handlers = [
            h for h in root.handlers if not getattr(h, _HANDLER_TAG, False)
        ]
        try:
            yield
        finally:
            root.handlers = saved
            root.setLevel(level)

    def test_silent_by_default(self, tmp_path, capsys):
        sso_main(_canonical_argv(tmp_path))
        captured = capsys.readouterr()
        assert captured.out == "", (
            f"expected silent CLI without -v, got: {captured.out!r}"
        )

    def test_verbose_emits_to_stdout(self, tmp_path, capsys):
        sso_main(_canonical_argv(tmp_path) + ["-v"])
        captured = capsys.readouterr()
        assert captured.out, "expected -v to produce stdout output"
        # Marker we know is present in the start line and end line.
        assert "[pproc.sso]" in captured.out
        # The pipeline logger should also be visible.
        assert "[pproc.climate.sso.pipeline]" in captured.out

    def test_double_verbose_emits_more(self, tmp_path, capsys):
        sso_main(_canonical_argv(tmp_path) + ["-v"])
        single = capsys.readouterr().out
        sso_main(_canonical_argv(tmp_path) + ["--verbose", "--verbose"])
        double = capsys.readouterr().out
        # DEBUG adds array-shape lines and stage-1 decision details on top
        # of every INFO line, so the byte volume should be strictly larger.
        assert len(double) > len(single), (
            f"expected -vv volume ({len(double)}) > -v volume ({len(single)})"
        )
