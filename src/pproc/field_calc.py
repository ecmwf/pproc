# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``pproc-field-calc``: a numpy-backed replacement for ``mir-compute``.

Reads one or more GRIB inputs, evaluates a formula (or a semicolon-separated
list of formulae) using :func:`pproc.climate.field_calc.evaluate_formula`,
and writes a single output GRIB file containing one message per
sub-formula.

The CLI is deliberately small: argparse only (no Conflator), in keeping
with the convention of other simple pproc tools such as ``pproc-interpol``
(see ``IMPLEMENTATIO_PLAN``). The :func:`main` entry point accepts an
optional ``argv`` list for unit testing without spawning subprocesses.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

from pproc.climate.field_calc import evaluate_formula, parse_variables
from pproc.common.io import decode_grib, decode_multi_grib, encode_grib


__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pproc-field-calc",
        description=(
            "Evaluate a mir-compute-style formula over GRIB inputs and write "
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
    # ``--metadata`` accepts one or more KEY=VAL tokens per invocation and
    # may be repeated. We use ``action='append'`` with ``nargs='+'`` and
    # rely on the argv pre-processor in :func:`_normalise_argv` to insert
    # an explicit terminator before trailing positional arguments — vanilla
    # argparse greedily consumes them otherwise (CPython issue #9338).
    parser.add_argument(
        "--metadata",
        nargs="+",
        action="append",
        default=None,
        metavar="KEY=VAL",
        help=(
            "GRIB metadata overrides for the output, e.g. "
            "--metadata shortName=sdor packingType=grid_simple. "
            "May be repeated."
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
    return parser


def _parse_metadata(groups: Optional[List[List[str]]]) -> dict:
    """Parse ``[[KEY=VAL, ...], ...]`` items into a dict of overrides.

    ``argparse`` with ``action='append', nargs='+'`` produces a list of
    lists (one inner list per ``--metadata`` invocation); flatten and
    parse each ``KEY=VAL`` token. The value side is left as a string;
    downstream ``construct_message`` knows how to coerce numeric/string
    GRIB keys.
    """
    if not groups:
        return {}
    out: dict = {}
    for group in groups:
        for item in group:
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


# Optional flags that take exactly one value (so we don't mis-classify their
# value as a "looks like a path" positional during argv normalisation).
_SINGLE_VALUE_FLAGS = {"--formula", "--variables", "--multi-dimensional"}


def _normalise_argv(argv: List[str]) -> List[str]:
    """Disambiguate ``--metadata K=V K=V ... INPUT ... OUTPUT``.

    ``argparse``'s positional-allocator (CPython issue #9338) cannot split
    a ``nargs='+'`` optional from a trailing ``nargs='+'`` positional when
    they sit next to each other on the command line. We walk argv and
    insert ``--`` immediately before the first non-``KEY=VAL`` token that
    follows a ``--metadata`` flag, which forces argparse to treat the
    remainder as positional arguments.

    The transformation is conservative: it only fires when ``--metadata``
    is in argv, and it never reorders user tokens.
    """
    if not argv:
        return list(argv)
    out: List[str] = []
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        out.append(tok)
        # Match either bare ``--metadata`` or ``--metadata=...`` (which
        # consumes only one value via the ``=`` syntax and needs no fixup).
        if tok == "--metadata":
            i += 1
            # Consume KEY=VAL tokens that follow.
            while i < n:
                nxt = argv[i]
                if nxt.startswith("--"):
                    break
                if "=" in nxt and not nxt.startswith("-"):
                    out.append(nxt)
                    i += 1
                    continue
                # First token without '=' that isn't a flag: this is the
                # start of the trailing positional arguments. Insert the
                # ``--`` separator so argparse stops consuming.
                if "--" not in out:
                    out.append("--")
                break
            continue
        # Skip the value of single-value optionals so we don't accidentally
        # try to interpret e.g. "--formula" "f1-f2" as something requiring
        # normalisation.
        if tok in _SINGLE_VALUE_FLAGS and i + 1 < n:
            out.append(argv[i + 1])
            i += 2
            continue
        i += 1
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
    """Entry point for the ``pproc-field-calc`` console script.

    Parameters
    ----------
    argv:
        Optional list of CLI arguments (excluding the program name).
        Defaults to ``sys.argv[1:]`` when ``None``.
    """
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_normalise_argv(raw_argv))

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

    # ----- evaluate each sub-formula and concatenate the encoded messages -----
    out_chunks: List[bytes] = []
    for sub in sub_formulae:
        result = evaluate_formula(sub, bindings)
        encoded = encode_grib(result, template_bytes, metadata=metadata)
        out_chunks.append(encoded)

    # ----- write output -----
    with open(output_path, "wb") as fh:
        for chunk in out_chunks:
            fh.write(chunk)


if __name__ == "__main__":  # pragma: no cover
    main()
