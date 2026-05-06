# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""End-to-end tests for the ``pproc-gradient`` CLI.

Drives :func:`pproc.gradient.main` directly (no subprocess) and compares
output against the reference intermediate produced by the legacy SSO ksh
pipeline (``orog_egrid_diff_grad`` at the climate-fields repo root).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pproc.gradient import main as gradient_main
from pproc.common.io import decode_grib, decode_multi_grib

# tests/ -> pproc/ -> pproc/ -> climate-fields/
REPO_ROOT = Path(__file__).resolve().parents[2]

_OROG_EGRID_DIFF = REPO_ROOT / "orog_egrid_diff"
_OROG_EGRID_DIFF_GRAD = REPO_ROOT / "orog_egrid_diff_grad"

_REQUIRED_DATA: tuple[Path, ...] = (
    _OROG_EGRID_DIFF,
    _OROG_EGRID_DIFF_GRAD,
)

_MISSING = tuple(p for p in _REQUIRED_DATA if not p.exists())

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=(
        "reference GRIB data not available at the workspace root; missing: "
        + ", ".join(str(p) for p in _MISSING)
    ),
)


class TestGradientCLI:
    def test_scalar_gradient_produces_two_messages(self, tmp_path):
        out = tmp_path / "grad.grib"
        gradient_main([str(_OROG_EGRID_DIFF), str(out)])
        msgs = decode_multi_grib(out.read_bytes(), 2)
        assert len(msgs) == 2

    def test_scalar_gradient_matches_reference(self, tmp_path):
        out = tmp_path / "grad.grib"
        gradient_main([str(_OROG_EGRID_DIFF), str(out)])
        actual_msgs = decode_multi_grib(out.read_bytes(), 2)
        ref_msgs = decode_multi_grib(_OROG_EGRID_DIFF_GRAD.read_bytes(), 2)
        for i, ((a_vals, _), (r_vals, _)) in enumerate(zip(actual_msgs, ref_msgs)):
            np.testing.assert_allclose(
                a_vals, r_vals, rtol=1e-5, equal_nan=True, err_msg=f"msg {i}"
            )

    def test_scalar_gradient_explicit_operation_flag(self, tmp_path):
        # Explicitly passing --operation scalar-gradient should yield same
        # bytes as the default invocation.
        out_default = tmp_path / "grad_default.grib"
        out_explicit = tmp_path / "grad_explicit.grib"
        gradient_main([str(_OROG_EGRID_DIFF), str(out_default)])
        gradient_main(
            [
                "--operation",
                "scalar-gradient",
                str(_OROG_EGRID_DIFF),
                str(out_explicit),
            ]
        )
        assert out_default.read_bytes() == out_explicit.read_bytes()

    def test_scalar_laplacian_produces_one_message(self, tmp_path):
        out = tmp_path / "lap.grib"
        gradient_main(
            [
                "--operation",
                "scalar-laplacian",
                str(_OROG_EGRID_DIFF),
                str(out),
            ]
        )
        # Single-message file: decode_grib should work without raising
        # "more than one message" (decode_grib reads the first only, but
        # we also assert via decode_multi_grib(_, 1) below).
        values, _meta = decode_grib(out.read_bytes())
        assert values.shape == (348528,)

        msgs = decode_multi_grib(out.read_bytes(), 1)
        assert len(msgs) == 1

    def test_no_poles_missing_values_flag(self, tmp_path):
        out_default = tmp_path / "grad_default.grib"
        out_no_poles = tmp_path / "grad_no_poles.grib"
        gradient_main([str(_OROG_EGRID_DIFF), str(out_default)])
        gradient_main(
            ["--no-poles-missing-values", str(_OROG_EGRID_DIFF), str(out_no_poles)]
        )
        # The reduced-Gaussian N256 input has no exact +/-90 deg points, so
        # the values themselves are not expected to differ; we just verify
        # both runs produce 2-message output successfully.
        assert len(decode_multi_grib(out_default.read_bytes(), 2)) == 2
        assert len(decode_multi_grib(out_no_poles.read_bytes(), 2)) == 2


class TestGradientCLIErrors:
    def test_missing_input_file(self, tmp_path):
        with pytest.raises((FileNotFoundError, SystemExit)):
            gradient_main(["/nonexistent/input.grib", str(tmp_path / "out.grib")])

    def test_invalid_operation(self, tmp_path):
        out = tmp_path / "out.grib"
        with pytest.raises(SystemExit):
            gradient_main(
                [
                    "--operation",
                    "not-a-real-op",
                    str(_OROG_EGRID_DIFF),
                    str(out),
                ]
            )

    def test_invalid_output_path(self):
        bad = "/nonexistent/directory/should/not/exist/out.grib"
        with pytest.raises((OSError, FileNotFoundError, SystemExit)):
            gradient_main([str(_OROG_EGRID_DIFF), bad])

    def test_missing_positional_args(self):
        # No INPUT or OUTPUT - argparse should bail.
        with pytest.raises(SystemExit):
            gradient_main([])

    def test_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            gradient_main(["--help"])
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        # Surface check that the help text mentions the key flags.
        assert "--operation" in captured.out
        assert "--no-poles-missing-values" in captured.out
