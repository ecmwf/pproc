# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``pproc-gradient``: thin CLI over :func:`pproc.climate.mir_ops.gradient`.

Reads a single-message GRIB input, computes a spatial gradient (or
Laplacian) via mir's nabla operator, and writes a GRIB output. For
``scalar-gradient`` the output contains two messages (in the order mir
emits them: dF/dlat first, then dF/dlon) — matching the byte layout of
the legacy ``orog_egrid_diff_grad`` intermediate. For ``scalar-laplacian``
the output contains a single message.

The CLI is deliberately small: argparse only (no Conflator), in keeping
with the convention of other simple pproc tools such as ``pproc-interpol``
and ``pproc-field-calc``. The :func:`main` entry point accepts an
optional ``argv`` list for unit testing without spawning subprocesses.

Implementation note: ``scalar-gradient`` delegates to
:func:`pproc.climate.mir_ops.gradient` (Unit D), which already handles
the BytesIO transport, typed-error wrapping, and 2-message split.
``scalar-laplacian`` is invoked inline via :class:`mir.Job` here rather
than extending ``mir_ops`` — keeping the climate operator surface
minimal as agreed in the WEAVE_PLAN. The pattern (BytesIO in/out plus
the ``nabla-poles-missing-values`` option) mirrors ``mir_ops.gradient``
so the two paths are consistent.
"""

from __future__ import annotations

import argparse
import io
from typing import List, Optional

import mir

from pproc.climate.mir_ops import (
    MirJobError,
    gradient as _mir_gradient,
)


__all__ = ["main"]


_VALID_OPERATIONS = ("scalar-gradient", "scalar-laplacian")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pproc-gradient",
        description=(
            "Compute the spatial gradient (or Laplacian) of a scalar GRIB "
            "field using mir's nabla operator. For scalar-gradient the "
            "output GRIB file contains two messages (dF/dlat, dF/dlon, in "
            "that order); for scalar-laplacian it contains one message."
        ),
    )
    parser.add_argument(
        "input",
        metavar="INPUT",
        help="Input GRIB file (single-message scalar field).",
    )
    parser.add_argument(
        "output",
        metavar="OUTPUT",
        help="Output GRIB file.",
    )
    parser.add_argument(
        "--operation",
        choices=_VALID_OPERATIONS,
        default="scalar-gradient",
        help=(
            "mir nabla operation to apply (default: %(default)s). "
            "scalar-gradient produces 2 output messages "
            "(dF/dlat, dF/dlon); scalar-laplacian produces 1."
        ),
    )
    parser.add_argument(
        "--no-poles-missing-values",
        dest="poles_missing_values",
        action="store_false",
        default=True,
        help=(
            "Disable flagging values at lat=+/-90 deg as missing in the "
            "output. Default behaviour (enabled) matches the legacy SSO "
            "ksh pipeline's --nabla-poles-missing-values option."
        ),
    )
    return parser


def _run_laplacian(grib_bytes: bytes, *, poles_missing_values: bool) -> bytes:
    """Run ``mir.Job(nabla='scalar-laplacian')`` on a single GRIB buffer.

    Mirrors the in-memory transport used by
    :func:`pproc.climate.mir_ops.gradient` (BytesIO in/out) but returns
    the raw single-message buffer mir emits without splitting.

    The mir bindings reject Python ``bool`` for boolean-typed options on
    at least some versions, so the ``poles_missing_values`` flag is
    coerced to the lowercase strings ``"true"``/``"false"`` that the
    underlying parametrisation accepts (matching ``mir_ops._coerce_mir_value``).
    """
    job = mir.Job(
        nabla="scalar-laplacian",
        nabla_poles_missing_values="true" if poles_missing_values else "false",
    )
    inp = io.BytesIO(grib_bytes)
    out = io.BytesIO()
    try:
        job.execute(inp, out)
    except RuntimeError as exc:
        raise MirJobError(f"gradient(nabla=scalar-laplacian) failed: {exc}") from exc
    return out.getvalue()


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Read input as bytes (raises FileNotFoundError on a missing file —
    # the CLI surface specifies that as acceptable error handling).
    with open(args.input, "rb") as f:
        grib_bytes = f.read()

    try:
        if args.operation == "scalar-gradient":
            gradx, grady = _mir_gradient(
                grib_bytes,
                poles_missing_values=args.poles_missing_values,
            )
            output_bytes = gradx + grady
        else:  # scalar-laplacian (argparse's choices restrict the values)
            output_bytes = _run_laplacian(
                grib_bytes,
                poles_missing_values=args.poles_missing_values,
            )
    except MirJobError as exc:
        parser.error(str(exc))

    with open(args.output, "wb") as f:
        f.write(output_bytes)


if __name__ == "__main__":  # pragma: no cover
    main()
