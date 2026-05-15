# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""End-to-end tests for the ``pproc-field-calc`` CLI.

These tests drive the ``main()`` entry point of :mod:`pproc.field_calc`
directly (no subprocess) using reference GRIB intermediates that live
at the climate-fields repo root.
"""

from pathlib import Path

import numpy as np
import pytest

from pproc.field_calc import main as field_calc_main
from pproc.common.io import decode_grib, decode_multi_grib

# tests/ -> pproc/ -> pproc/ -> climate-fields/
REPO_ROOT = Path(__file__).resolve().parents[2]


# Pre-compute the paths and skip the whole module if the reference fixtures
# are missing (e.g. running outside the climate-fields working tree).
_OROG_5KM = REPO_ROOT / "data" / "input" / "ifs" / "orog_5km"
_OROG_EGRID_N2000 = REPO_ROOT / "orog_egrid_N256"
_OROG_EGRID_DIFF = REPO_ROOT / "orog_egrid_diff"
_OROG_EGRID_DIFF_GRAD = REPO_ROOT / "orog_egrid_diff_grad"
_OROG_EGRID_DIFF_GRADXY = REPO_ROOT / "orog_egrid_diff_gradxy"
_OROG_MGRID_DIFF_SQ = REPO_ROOT / "orog_mgrid_diff_sq"
_LAND_MASK = REPO_ROOT / "data" / "input" / "ifs" / "land_mask"
_STDGWD_REF = REPO_ROOT / "data" / "output" / "stdgwd"

_REQUIRED_DATA: tuple[Path, ...] = (
    _OROG_5KM,
    _OROG_EGRID_N2000,
    _OROG_EGRID_DIFF,
    _OROG_EGRID_DIFF_GRAD,
    _OROG_EGRID_DIFF_GRADXY,
    _OROG_MGRID_DIFF_SQ,
    _LAND_MASK,
    _STDGWD_REF,
)

_MISSING = tuple(p for p in _REQUIRED_DATA if not p.exists())

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=(
        "reference GRIB data not available at the workspace root; missing: "
        + ", ".join(str(p) for p in _MISSING)
    ),
)


class TestFieldCalcCLI:
    def test_two_input_subtraction(self, tmp_path):
        out = tmp_path / "diff.grib"
        field_calc_main(
            [
                "--formula",
                "f1 - f2",
                str(_OROG_5KM),
                str(_OROG_EGRID_N2000),
                str(out),
            ]
        )
        actual, _ = decode_grib(out.read_bytes())
        expected, _ = decode_grib(_OROG_EGRID_DIFF.read_bytes())
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_named_variables_subtraction(self, tmp_path):
        out = tmp_path / "diff.grib"
        field_calc_main(
            [
                "--variables",
                "a;b",
                "--formula",
                "a - b",
                str(_OROG_5KM),
                str(_OROG_EGRID_N2000),
                str(out),
            ]
        )
        actual, _ = decode_grib(out.read_bytes())
        expected, _ = decode_grib(_OROG_EGRID_DIFF.read_bytes())
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_multi_dimensional_input_two_messages(self, tmp_path):
        out = tmp_path / "cross.grib"
        field_calc_main(
            [
                "--variables",
                "gx;gy",
                "--formula",
                "gx*gy",
                "--multi-dimensional",
                "2",
                str(_OROG_EGRID_DIFF_GRAD),
                str(out),
            ]
        )
        actual, _ = decode_grib(out.read_bytes())
        expected, _ = decode_grib(_OROG_EGRID_DIFF_GRADXY.read_bytes())
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_metadata_override(self, tmp_path):
        out = tmp_path / "out.grib"
        field_calc_main(
            [
                "--variables",
                "var",
                "--formula",
                "sqrt(var)",
                "--metadata",
                "shortName=sdor",
                "packingType=grid_simple",
                str(_OROG_MGRID_DIFF_SQ),
                str(out),
            ]
        )
        _, meta = decode_grib(out.read_bytes())
        assert meta["shortName"] == "sdor"
        assert meta["packingType"] == "grid_simple"

    def test_default_variables_f1_f2(self, tmp_path):
        out = tmp_path / "diff.grib"
        field_calc_main(
            [
                "--formula",
                "f1 - f2",
                str(_OROG_5KM),
                str(_OROG_EGRID_N2000),
                str(out),
            ]
        )
        result, _ = decode_grib(out.read_bytes())
        assert result.shape == (348528,)
        # And it should equal the reference difference.
        expected, _ = decode_grib(_OROG_EGRID_DIFF.read_bytes())
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_multiple_sub_formulae(self, tmp_path):
        out = tmp_path / "out.grib"
        field_calc_main(
            [
                "--variables",
                "a;b",
                "--formula",
                "a-b;a+b",
                str(_OROG_5KM),
                str(_OROG_EGRID_N2000),
                str(out),
            ]
        )
        msgs = decode_multi_grib(out.read_bytes(), 2)
        assert len(msgs) == 2

        a_arr, _ = decode_grib(_OROG_5KM.read_bytes())
        b_arr, _ = decode_grib(_OROG_EGRID_N2000.read_bytes())
        np.testing.assert_allclose(msgs[0][0], a_arr - b_arr, rtol=1e-5)
        np.testing.assert_allclose(msgs[1][0], a_arr + b_arr, rtol=1e-5)

    def test_three_sub_formulae(self, tmp_path):
        out = tmp_path / "out.grib"
        field_calc_main(
            [
                "--variables",
                "a;b",
                "--formula",
                "a-b;a+b;sqrt(a^2+b^2)",
                str(_OROG_5KM),
                str(_OROG_EGRID_N2000),
                str(out),
            ]
        )
        msgs = decode_multi_grib(out.read_bytes(), 3)
        assert len(msgs) == 3

    def test_reference_stdgwd_reproduction(self, tmp_path):
        out = tmp_path / "stdgwd.grib"
        field_calc_main(
            [
                "--variables",
                "var;lsm",
                "--formula",
                "sqrt(var) * lsm",
                "--metadata",
                "shortName=sdor",
                "packingType=grid_simple",
                str(_OROG_MGRID_DIFF_SQ),
                str(_LAND_MASK),
                str(out),
            ]
        )
        actual, meta = decode_grib(out.read_bytes())
        expected, _ = decode_grib(_STDGWD_REF.read_bytes())
        # As noted in the task spec: float32-vs-float64 D-F1 drift on stdgwd.
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
        assert meta["shortName"] == "sdor"
        assert meta["packingType"] == "grid_simple"


class TestFieldCalcCLIErrors:
    def test_missing_formula_arg(self):
        with pytest.raises(SystemExit):
            field_calc_main([str(_OROG_5KM), "/tmp/does_not_matter.grib"])

    def test_multi_dim_with_multiple_inputs_rejects(self, tmp_path):
        out = tmp_path / "out.grib"
        with pytest.raises(SystemExit):
            field_calc_main(
                [
                    "--formula",
                    "f1-f2",
                    "--multi-dimensional",
                    "2",
                    str(_OROG_5KM),
                    str(_OROG_EGRID_N2000),
                    str(out),
                ]
            )

    def test_variable_count_mismatch(self, tmp_path):
        # 2 inputs but 3 declared variables.
        out = tmp_path / "out.grib"
        with pytest.raises(SystemExit):
            field_calc_main(
                [
                    "--variables",
                    "a;b;c",
                    "--formula",
                    "a-b",
                    str(_OROG_5KM),
                    str(_OROG_EGRID_N2000),
                    str(out),
                ]
            )

    def test_multi_dim_variable_count_mismatch(self, tmp_path):
        # multi-dimensional 2 declared but only 1 variable name.
        out = tmp_path / "out.grib"
        with pytest.raises(SystemExit):
            field_calc_main(
                [
                    "--variables",
                    "gx",
                    "--formula",
                    "gx",
                    "--multi-dimensional",
                    "2",
                    str(_OROG_EGRID_DIFF_GRAD),
                    str(out),
                ]
            )

    def test_undefined_variable_in_formula(self, tmp_path):
        out = tmp_path / "out.grib"
        with pytest.raises((NameError, SystemExit)):
            field_calc_main(
                [
                    "--variables",
                    "a;b",
                    "--formula",
                    "a - c",  # `c` is undefined
                    str(_OROG_5KM),
                    str(_OROG_EGRID_N2000),
                    str(out),
                ]
            )

    def test_invalid_output_path(self, tmp_path):
        # Parent directory does not exist.
        bad = "/nonexistent/directory/should/not/exist/out.grib"
        with pytest.raises((OSError, FileNotFoundError, SystemExit)):
            field_calc_main(
                [
                    "--formula",
                    "f1 - f2",
                    str(_OROG_5KM),
                    str(_OROG_EGRID_N2000),
                    bad,
                ]
            )

    def test_no_inputs(self):
        # OUTPUT only with no INPUTs - argparse requires at least 1 input + 1 output.
        with pytest.raises(SystemExit):
            field_calc_main(["--formula", "f1", "/tmp/out.grib"])

    def test_metadata_without_equals_rejected(self, tmp_path):
        out = tmp_path / "out.grib"
        with pytest.raises(SystemExit):
            field_calc_main(
                [
                    "--variables",
                    "var",
                    "--formula",
                    "sqrt(var)",
                    "--metadata",
                    "shortName_no_equals",
                    str(_OROG_MGRID_DIFF_SQ),
                    str(out),
                ]
            )


class TestFieldCalcCLIVerbose:
    """Smoke tests for the ``-v`` / ``--verbose`` count flag.

    See :class:`TestGradientCLIVerbose` for the rationale (silent by
    default, monotonic increase in stdout with verbosity).
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
        out = tmp_path / "diff.grib"
        field_calc_main(
            [
                "--formula",
                "f1 - f2",
                str(_OROG_5KM),
                str(_OROG_EGRID_N2000),
                str(out),
            ]
        )
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_emits_to_stdout(self, tmp_path, capsys):
        out = tmp_path / "diff.grib"
        field_calc_main(
            [
                "-v",
                "--formula",
                "f1 - f2",
                str(_OROG_5KM),
                str(_OROG_EGRID_N2000),
                str(out),
            ]
        )
        captured = capsys.readouterr()
        assert captured.out, "expected -v to produce stdout output"
        assert "[pproc.field_calc]" in captured.out
