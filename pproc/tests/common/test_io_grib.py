# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the GRIB codec helpers in :mod:`pproc.common.io`.

Covers ``decode_grib``, ``decode_grib_with_metadata``, ``decode_multi_grib``
and ``encode_grib``. Reference inputs are the SSO weave intermediates at the
repository root: ``orog_egrid`` (single message, N48), ``orog_egrid_diff_grad``
(2 messages, N256) and ``KLMLprime_lsm`` (5 messages, N256).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pproc.common import io


# Repository root: pproc/pproc/tests/common/test_io_grib.py -> ../../../..
REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

OROG_EGRID = os.path.join(REPO_ROOT, "orog_egrid")
OROG_EGRID_DIFF_GRAD = os.path.join(REPO_ROOT, "orog_egrid_diff_grad")
KLMLPRIME_LSM = os.path.join(REPO_ROOT, "KLMLprime_lsm")


_REQUIRED_DATA: tuple[str, ...] = (
    OROG_EGRID,
    OROG_EGRID_DIFF_GRAD,
    KLMLPRIME_LSM,
)

_MISSING = tuple(p for p in _REQUIRED_DATA if not os.path.exists(p))

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=(
        "reference GRIB data not available at the workspace root; missing: "
        + ", ".join(_MISSING)
    ),
)


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# decode_grib
# ---------------------------------------------------------------------------


def test_decode_grib_single_message_orog_egrid():
    buf = _read(OROG_EGRID)
    values, meta = io.decode_grib(buf)

    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float64
    assert values.shape == (13280,)
    assert isinstance(meta, dict)

    # Required metadata keys (per task spec).
    for key in (
        "gridType",
        "gridName",
        "numberOfDataPoints",
        "shortName",
        "packingType",
        "edition",
        "paramId",
        "bitsPerValue",
    ):
        assert key in meta, f"missing metadata key: {key}"

    assert meta["gridName"] == "N48"
    assert meta["numberOfDataPoints"] == 13280
    assert meta["gridType"] == "reduced_gg"


def test_decode_grib_with_metadata_returns_gribmetadata():
    buf = _read(OROG_EGRID)
    values, gm = io.decode_grib_with_metadata(buf)

    assert isinstance(gm, io.GribMetadata)
    assert isinstance(values, np.ndarray)
    assert values.shape == (13280,)

    # The dict-form keys should be reachable on the GribMetadata.
    _, meta = io.decode_grib(buf)
    for key in (
        "gridType",
        "gridName",
        "numberOfDataPoints",
        "shortName",
        "packingType",
        "edition",
        "paramId",
    ):
        assert gm.get(key) == meta[key]


# ---------------------------------------------------------------------------
# decode_multi_grib
# ---------------------------------------------------------------------------


def test_decode_multi_grib_orog_egrid_diff_grad_two_messages():
    buf = _read(OROG_EGRID_DIFF_GRAD)
    msgs = io.decode_multi_grib(buf, 2)

    assert isinstance(msgs, list)
    assert len(msgs) == 2
    for values, meta in msgs:
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float64
        assert values.shape == (348528,)
        assert meta["gridName"] == "N256"
        assert meta["numberOfDataPoints"] == 348528


def test_decode_multi_grib_klmlprime_lsm_five_messages():
    buf = _read(KLMLPRIME_LSM)
    msgs = io.decode_multi_grib(buf, 5)

    assert len(msgs) == 5
    for values, meta in msgs:
        assert values.shape == (348528,)
        assert meta["gridName"] == "N256"


def test_decode_multi_grib_too_many_requested_raises():
    buf = _read(OROG_EGRID_DIFF_GRAD)
    with pytest.raises(ValueError):
        io.decode_multi_grib(buf, 5)


# ---------------------------------------------------------------------------
# encode_grib (round-trip)
# ---------------------------------------------------------------------------


def test_encode_decode_round_trip_bit_identical_values():
    buf = _read(OROG_EGRID)
    values, meta = io.decode_grib(buf)

    encoded = io.encode_grib(values, buf)
    values2, meta2 = io.decode_grib(encoded)

    assert values2.shape == values.shape
    # Same packing + same template => bit-identical recovery.
    np.testing.assert_array_equal(values2, values)

    # Critical template metadata keys must survive the round trip.
    for key in (
        "gridType",
        "gridName",
        "numberOfDataPoints",
        "shortName",
        "packingType",
        "edition",
        "paramId",
        "bitsPerValue",
    ):
        assert meta2[key] == meta[key], (
            f"key {key} drifted: {meta2[key]} != {meta[key]}"
        )


def test_encode_grib_accepts_gribmetadata_template():
    """Pattern decision D-A: encode_grib is polymorphic on its template."""
    buf = _read(OROG_EGRID_DIFF_GRAD)
    msgs = io.decode_multi_grib(buf, 2)
    values_first, _ = msgs[0]
    _, gm_second = io.decode_grib_with_metadata(
        io.encode_grib(msgs[1][0], buf)  # bytes round-trip to obtain wire bytes
    )

    # Use the second message's GribMetadata as a template, write fresh values.
    fresh = np.linspace(0.0, 1.0, num=values_first.shape[0], dtype=np.float64)
    encoded = io.encode_grib(fresh, gm_second)

    out_values, out_meta = io.decode_grib(encoded)
    assert out_values.shape == fresh.shape
    assert out_meta["gridName"] == "N256"
    assert out_meta["numberOfDataPoints"] == 348528


# ---------------------------------------------------------------------------
# NaN / missing-value bitmap round-trip
# ---------------------------------------------------------------------------


def test_nan_missing_round_trip_preserves_nan_positions():
    buf = _read(OROG_EGRID)
    values, _ = io.decode_grib(buf)

    nan_idx = np.array([0, 17, 1234, 9999])
    values_with_nan = values.copy()
    values_with_nan[nan_idx] = np.nan

    encoded = io.encode_grib(values_with_nan, buf)
    decoded, _ = io.decode_grib(encoded)

    assert decoded.shape == values_with_nan.shape
    # NaN positions must survive
    assert np.all(np.isnan(decoded[nan_idx]))
    # Non-NaN positions remain finite (we don't assert bit-identical here
    # because flipping bitmapPresent on can renormalise packing references).
    finite_mask = np.ones_like(decoded, dtype=bool)
    finite_mask[nan_idx] = False
    assert np.all(np.isfinite(decoded[finite_mask]))


# ---------------------------------------------------------------------------
# Metadata override
# ---------------------------------------------------------------------------


def test_encode_grib_metadata_override_short_name_and_packing():
    buf = _read(OROG_EGRID)
    values, _ = io.decode_grib(buf)

    encoded = io.encode_grib(
        values,
        buf,
        {"shortName": "sdor", "packingType": "grid_simple"},
    )
    _, meta = io.decode_grib(encoded)

    assert meta["shortName"] == "sdor"
    assert meta["packingType"] == "grid_simple"
