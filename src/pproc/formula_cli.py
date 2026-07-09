# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``pproc-formula``: evaluate arithmetic formulae over GRIB fields.

Reads one or more GRIB inputs, evaluates a formula (or a semicolon-separated
list of formulae) using :func:`pproc.formula.evaluate_formula`,
and writes a single output GRIB file containing one message per
sub-formula.

The CLI is deliberately small: argparse only (no Conflator), in keeping
with the convention of other simple pproc tools such as ``pproc-interpol``.
The :func:`main` entry point accepts an optional ``argv`` list for unit
testing without spawning subprocesses.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import List, Optional, Tuple

from pproc.climate._logging import configure_logging
from pproc.common.io import decode_grib, decode_multi_grib, encode_grib
from pproc.formula import evaluate_formula, parse_variables


__all__ = ["main"]


logger = logging.getLogger("pproc.formula")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pproc-formula",
        description=(
            "Evaluate an arithmetic formula over GRIB inputs and write "
            "a GRIB output (one message per sub-formula)."
        ),
    )
    parser.add_argument(
        "--formula",
        required=True,
        help=(
            "Formula expression. May contain ';' to separate multiple "
            "sub-formulae; each sub-formula produces one output message."
        ),
    )
    parser.add_argument(
        "--variables",
        default=None,
        help=(
            "Semicolon-separated list of variable names in input order. "
            "Defaults to f1, f2, ... fN."
        ),
    )
    parser.add_argument(
        "--multi-dimensional",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If given, treat the (single) input file as N consecutive GRIB "
            "messages. Incompatible with multiple input files."
        ),
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=None,
        metavar="KEY=VAL",
        help=(
            "GRIB metadata override for the output as a single KEY=VAL, e.g. "
            "--metadata shortName=sdor. Repeat the flag for multiple "
            "overrides: --metadata shortName=sdor --metadata packingType=grid_simple."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="INPUT_OR_OUTPUT",
        help=(
            "One or more INPUT GRIB files followed by a single OUTPUT GRIB "
            "file. With --multi-dimensional N, exactly one INPUT is allowed."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Increase logging verbosity to stdout: -v shows INFO "
            "(per-sub-formula evaluation); -vv shows DEBUG. "
            "Absent: silent (WARNING)."
        ),
    )
    return parser


def _parse_metadata(items: Optional[List[str]]) -> dict:
    """Parse ``[KEY=VAL, ...]`` items into a dict of overrides.

    Each ``--metadata`` occurrence contributes one ``KEY=VAL`` token.
    The value side is left as a string; downstream ``construct_message``
    coerces numeric/string GRIB keys as needed.
    """
    if not items:
        return {}
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"--metadata entry {item!r} is missing '=' (expected KEY=VAL)"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if key == "":
            raise ValueError(f"--metadata entry {item!r} has an empty key")
        out[key] = value
    return out


def _split_formulae(formula: str) -> List[str]:
    parts = [seg.strip() for seg in formula.split(";")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("--formula must contain at least one sub-formula")
    return parts


def _split_inputs_output(paths: List[str]) -> Tuple[List[str], str]:
    if len(paths) < 2:
        raise ValueError(
            "expected at least one INPUT GRIB and one OUTPUT GRIB path "
            f"(got {len(paths)})"
        )
    return paths[:-1], paths[-1]


def _load_inputs(
    input_paths: List[str], multi_dim: Optional[int]
) -> Tuple[list, bytes]:
    """Return (list of value arrays, template bytes).

    The template bytes are the wire bytes of the *first* input message
    (the multi-dimensional case uses the first message of the file).
    """
    if multi_dim is not None:
        if len(input_paths) != 1:
            raise ValueError(
                "--multi-dimensional is only valid with a single input file; "
                f"got {len(input_paths)} input files"
            )
        with open(input_paths[0], "rb") as fh:
            buf = fh.read()
        msgs = decode_multi_grib(buf, multi_dim)
        arrays = [m[0] for m in msgs]
        template_bytes = buf  # encode_grib reads only the first message
        return arrays, template_bytes

    arrays = []
    template_bytes: Optional[bytes] = None
    for path in input_paths:
        with open(path, "rb") as fh:
            buf = fh.read()
        if template_bytes is None:
            template_bytes = buf
        values, _ = decode_grib(buf)
        arrays.append(values)
    assert template_bytes is not None  # guaranteed by len(input_paths) >= 1
    return arrays, template_bytes


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the ``pproc-formula`` console script.

    Parameters
    ----------
    argv:
        Optional list of CLI arguments (excluding the program name).
        Defaults to ``sys.argv[1:]`` when ``None``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    # ----- validate / parse the structural arguments -----
    try:
        input_paths, output_path = _split_inputs_output(args.paths)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        sub_formulae = _split_formulae(args.formula)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        metadata = _parse_metadata(args.metadata)
    except ValueError as exc:
        parser.error(str(exc))

    # Determine the variable names. With --multi-dimensional N, the
    # *number of arrays* coming out of the single input file is N, so the
    # variable count must match N (not the number of input paths).
    if args.multi_dimensional is not None:
        if args.multi_dimensional <= 0:
            parser.error(
                f"--multi-dimensional must be a positive integer, "
                f"got {args.multi_dimensional}"
            )
        if len(input_paths) != 1:
            parser.error(
                "--multi-dimensional is only valid with exactly one INPUT "
                f"file; got {len(input_paths)}"
            )
        n_arrays = args.multi_dimensional
    else:
        n_arrays = len(input_paths)

    if args.variables is None:
        var_names = [f"f{i + 1}" for i in range(n_arrays)]
    else:
        try:
            var_names = parse_variables(args.variables)
        except ValueError as exc:
            parser.error(str(exc))
        if len(var_names) != n_arrays:
            parser.error(
                f"--variables declares {len(var_names)} name(s) "
                f"({var_names}) but {n_arrays} field(s) "
                f"{'are produced by --multi-dimensional' if args.multi_dimensional else 'were supplied as inputs'}"
            )

    # ----- load inputs -----
    try:
        arrays, template_bytes = _load_inputs(input_paths, args.multi_dimensional)
    except (OSError, ValueError) as exc:
        parser.error(f"failed to read input GRIB: {exc}")

    bindings = dict(zip(var_names, arrays))

    start_msg = (
        "pproc-formula start"
        f" formula={args.formula!r}"
        f" variables={var_names}"
        f" inputs={input_paths}"
        f" output={output_path}"
        f" multi_dimensional={args.multi_dimensional}"
    )
    logger.info(start_msg)
    t0 = time.monotonic()

    # ----- evaluate each sub-formula and concatenate the encoded messages -----
    out_chunks: List[bytes] = []
    total = len(sub_formulae)
    for i, sub in enumerate(sub_formulae, start=1):
        logger.info("evaluating formula %d/%d: %s", i, total, sub)
        result = evaluate_formula(sub, bindings)
        encoded = encode_grib(result, template_bytes, metadata=metadata)
        out_chunks.append(encoded)

    # ----- write output -----
    with open(output_path, "wb") as fh:
        for chunk in out_chunks:
            fh.write(chunk)

    logger.info("pproc-formula done elapsed=%.3f", time.monotonic() - t0)


if __name__ == "__main__":  # pragma: no cover
    main()
