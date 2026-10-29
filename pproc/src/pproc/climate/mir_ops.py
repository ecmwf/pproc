# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Numpy-friendly wrappers around :class:`mir.Job`.

This module exposes thin, in-memory wrappers around three mir operations the
SSO climate-fields pipeline relies on:

* :func:`interpolate` — gridded interpolation (conservative, bilinear,
  nearest-neighbour, ...).
* :func:`gradient` — scalar gradient via ``mir.Job(nabla='scalar-gradient')``,
  returning the two output messages (∂f/∂lat, ∂f/∂lon) as separate single-
  message GRIB byte buffers.

Transport between mir and the caller is GRIB bytes via :class:`io.BytesIO` —
no temporary files are created or left behind. See the Tier 2 notes in
``.weave/learnings/sso-migration.md`` for the design context (Pattern decision
to drive all transport through Unit A's codec, and the D-K1 reference-match
acceptance criterion).
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any, Dict, Optional, Tuple

import eccodes
import mir


__all__ = [
    "MirJobError",
    "MirInvalidArgument",
    "interpolate",
    "gradient",
]


logger = logging.getLogger(__name__)


def _decode_point_count(grib_bytes: bytes) -> Optional[int]:
    """Return ``numberOfDataPoints`` of the first message, or ``None`` on error.

    Used only for INFO-level log lines; failures are silently mapped to
    ``None`` so logging never raises a user-visible error.
    """
    handle = None
    try:
        reader = eccodes.MemoryReader(grib_bytes)
        msg = next(iter(reader), None)
        if msg is None:
            return None
        # earthkit/eccodes message API: ``get`` retrieves a single key.
        return int(msg.get("numberOfDataPoints"))
    except Exception:  # noqa: BLE001 — logging must never raise
        return None
    finally:
        # ``MemoryReader`` does not need explicit close, but be defensive.
        del handle


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------


class MirJobError(RuntimeError):
    """Raised when a :class:`mir.Job` execution fails for any reason.

    Wraps the underlying :class:`RuntimeError` thrown by the mir Python
    bindings so callers can catch a typed exception (and walk to the
    original via :pyattr:`__cause__`).
    """


class MirInvalidArgument(MirJobError, ValueError):
    """Raised when an invalid mir argument (method, grid, ...) is supplied.

    Subclasses :class:`ValueError` so callers that already filter on the
    standard ``ValueError`` hierarchy continue to work.
    """


# Patterns for narrowing a generic mir RuntimeError into ``MirInvalidArgument``.
# The mir bindings raise ``RuntimeError`` with messages like:
#
#   "Serious bug: MethodFactory: unknown 'not-a-real-method'"
#   "GridSpec: cannot decode 'not-a-grid'"
#
# These map to user-facing argument errors (an unknown method or grid spec),
# whereas other RuntimeErrors (out of memory, library bugs, ...) should
# surface as :class:`MirJobError` so the operator can tell them apart.
_INVALID_ARGUMENT_MARKERS: Tuple[re.Pattern, ...] = (
    re.compile(r"MethodFactory:\s*unknown", re.IGNORECASE),
    re.compile(r"unknown\s+(?:method|interpolation|grid|nabla)", re.IGNORECASE),
    re.compile(r"cannot\s+decode", re.IGNORECASE),
    re.compile(r"invalid\s+(?:grid|method|spec)", re.IGNORECASE),
    re.compile(r"unsupported", re.IGNORECASE),
    re.compile(r"Cannot find a Grid", re.IGNORECASE),
)


def _wrap_mir_error(exc: BaseException, context: str) -> MirJobError:
    """Translate a low-level mir/eckit RuntimeError into a typed exception."""
    text = str(exc)
    for marker in _INVALID_ARGUMENT_MARKERS:
        if marker.search(text):
            return MirInvalidArgument(f"{context}: {text}")
    return MirJobError(f"{context}: {text}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_bytes_like(grib_bytes: Any, name: str = "grib_bytes") -> bytes:
    """Validate and coerce a bytes-like input to plain ``bytes``."""
    if not isinstance(grib_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like, got {type(grib_bytes).__name__}")
    return bytes(grib_bytes)


def _coerce_mir_value(value: Any) -> Any:
    """Coerce Python values to forms the mir bindings accept.

    The mir bindings reject Python ``bool`` for boolean-typed options on at
    least some versions (it raises "Cannot convert 1 from int to bool"), so
    we map Python booleans to the lowercase strings ``"true"``/``"false"``
    that the underlying parametrisation accepts. All other values pass
    through unchanged.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _build_job(**options: Any) -> mir.Job:
    """Construct a :class:`mir.Job` with the given (Pythonic) options.

    Keys with underscores are converted to hyphens automatically by the mir
    bindings, so callers can pass e.g. ``nabla_poles_missing_values=True``
    and have it become ``nabla-poles-missing-values=true`` on the wire.
    """
    coerced = {key: _coerce_mir_value(val) for key, val in options.items()}
    try:
        return mir.Job(**coerced)
    except RuntimeError as exc:
        raise _wrap_mir_error(exc, "mir.Job construction failed") from exc


def _execute_job(job: mir.Job, grib_bytes: bytes, *, context: str) -> bytes:
    """Run a prepared :class:`mir.Job` over an in-memory GRIB buffer."""
    inp = io.BytesIO(grib_bytes)
    out = io.BytesIO()
    try:
        job.execute(inp, out)
    except RuntimeError as exc:
        raise _wrap_mir_error(exc, context) from exc
    return out.getvalue()


def _split_messages(grib_bytes: bytes) -> list[bytes]:
    """Split a multi-message GRIB buffer into per-message byte buffers.

    Uses :class:`eccodes.MemoryReader` to walk the buffer and
    :meth:`eccodes.Message.get_buffer` to extract each message's wire bytes
    verbatim — no decode/encode round-trip, so the output is byte-exact.
    """
    reader = eccodes.MemoryReader(grib_bytes)
    return [message.get_buffer() for message in reader]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def interpolate(
    grib_bytes: bytes,
    *,
    grid: str,
    method: str,
    **kwargs: Any,
) -> bytes:
    """Interpolate a GRIB field via :class:`mir.Job`, returning GRIB bytes.

    Parameters
    ----------
    grib_bytes:
        Input GRIB message(s) as bytes.
    grid:
        Target grid spec, e.g. ``"N48"``, ``"N256"``, ``"O1280"``,
        ``"1.0/1.0"``.
    method:
        Interpolation method. The full list lives in mir; common values are
        ``"grid-box-average"``, ``"structured-bilinear"``,
        ``"nearest-neighbour"``, ``"nearest-lsm"``.
    **kwargs:
        Additional :class:`mir.Job` options. Underscores in keys convert to
        hyphens (e.g. ``lsm_selection`` → ``lsm-selection``); Python bools
        are converted to the lowercase strings the mir bindings accept.

    Returns
    -------
    bytes
        Output GRIB message bytes on the target grid.

    Raises
    ------
    MirInvalidArgument
        If ``method`` or ``grid`` is unrecognised by mir.
    MirJobError
        For any other failure surfaced by mir.
    TypeError
        If ``grib_bytes`` is not bytes-like.
    """
    payload = _ensure_bytes_like(grib_bytes)

    options: Dict[str, Any] = {"grid": grid, "interpolation": method}
    options.update(kwargs)

    job = _build_job(**options)
    result = _execute_job(
        job,
        payload,
        context=f"interpolate(grid={grid!r}, method={method!r}) failed",
    )
    # Operator-visibility line. Decoding ``numberOfDataPoints`` is cheap
    # (header read only) and runs only when INFO is enabled, so the
    # overhead is bounded to verbose runs.
    if logger.isEnabledFor(logging.INFO):
        n_in = _decode_point_count(payload)
        n_out = _decode_point_count(result)
        logger.info(
            "interpolate grid=%s method=%s input=%s pts → output=%s pts",
            grid,
            method,
            n_in if n_in is not None else "?",
            n_out if n_out is not None else "?",
        )
    return result


def gradient(
    grib_bytes: bytes,
    *,
    poles_missing_values: bool = True,
) -> Tuple[bytes, bytes]:
    """Compute the scalar gradient (∂f/∂lat, ∂f/∂lon) via mir's nabla.

    Parameters
    ----------
    grib_bytes:
        Input GRIB message bytes (single scalar field).
    poles_missing_values:
        If ``True`` (default), values at lat=±90° in the output gradient are
        flagged as missing — equivalent to the mir option
        ``nabla-poles-missing-values=true`` used by the legacy SSO ksh
        pipeline. Note: for reduced Gaussian grids whose first/last
        latitude row sits *just inside* ±90° (e.g. N256 → ±89.731°), no
        points are eligible to be flagged, so the resulting buffer carries
        no missing entries.

    Returns
    -------
    tuple of (bytes, bytes)
        ``(gradx_bytes, grady_bytes)`` — two single-message GRIB buffers,
        in the order produced by mir (∂f/∂lat first, then ∂f/∂lon),
        matching the byte layout of the legacy ``orog_egrid_diff_grad``
        intermediate. Each buffer is byte-exact with mir's output (no
        decode/re-encode round-trip).

    Raises
    ------
    MirJobError
        On any mir execution failure.
    TypeError
        If ``grib_bytes`` is not bytes-like.
    ValueError
        If mir does not produce exactly two output messages.
    """
    payload = _ensure_bytes_like(grib_bytes)

    options: Dict[str, Any] = {
        "nabla": "scalar-gradient",
        "nabla_poles_missing_values": poles_missing_values,
    }
    job = _build_job(**options)
    if logger.isEnabledFor(logging.INFO):
        n_in = _decode_point_count(payload)
        logger.info(
            "gradient poles_missing_values=%s input=%s pts → 2 messages",
            poles_missing_values,
            n_in if n_in is not None else "?",
        )
    multi = _execute_job(
        job,
        payload,
        context="gradient(nabla=scalar-gradient) failed",
    )

    parts = _split_messages(multi)
    if len(parts) != 2:
        raise ValueError(
            f"expected 2 messages from mir nabla scalar-gradient, got {len(parts)}"
        )
    return parts[0], parts[1]
