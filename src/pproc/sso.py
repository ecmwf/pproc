# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``pproc-sso``: monolithic CLI driving the SSO pipeline end-to-end.

Reads inputs (orography GRIB + land-mask GRIB + grid configuration), invokes
:func:`pproc.climate.sso.pipeline.compute_sso`, and writes the four output
GRIB files (``stdgwd``, ``slogwd``, ``anggwd``, ``isogwd``) to the chosen
output directory.

The CLI is deliberately small (argparse only, no Conflator) in keeping with
the convention of other simple ``pproc`` tools such as ``pproc-interpol``,
``pproc-gradient``, and ``pproc-field-calc``. The :func:`main` entry point
accepts an optional ``argv`` list for unit testing without spawning
subprocesses.

YAML config support
-------------------
``--config FILE`` loads a YAML document via ``yaml.safe_load`` and merges its
fields into the parsed argparse namespace. CLI arguments take precedence:
any value the user passes on the command line overrides the corresponding
YAML field. The YAML keys must match the snake-case
:class:`SSOConfig` field names (``orography``, ``land_mask``,
``target_grid``, ``model_grid_type``, ``model_resolution``, ``output_grid``,
``effective_resolution``, ``source_orography``, ``output_dir``,
``grib_roundtrip``, ``dump_intermediates``).

Boolean flags (``--grib-roundtrip``, ``--dump-intermediates``) use
``action="store_true"`` and default to ``False``. They can be enabled either
on the CLI or via YAML; an explicit ``True`` on the CLI cannot be overridden
back to ``False`` by YAML (this is the intended asymmetry — YAML supplies
defaults, CLI overrides them).

Filesystem safety
-----------------
``--output-dir`` is treated as a normal path supplied by the user; no path-
traversal protection is applied. The 16 intermediate filenames written under
``--dump-intermediates`` are hard-coded inside :mod:`pproc.climate.sso.pipeline`
(see ``_write_intermediate``); none of them is user-controlled. The output
directory is created on demand.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, List, Optional

import yaml

from pproc.climate.sso.config import SSOConfig
from pproc.climate.sso.pipeline import compute_sso


__all__ = ["main"]


# Names of the four output GRIB files written under ``--output-dir``. Order
# is documented but not load-bearing — the dict returned by ``compute_sso``
# is keyed by these same names.
_OUTPUT_NAMES = ("stdgwd", "slogwd", "anggwd", "isogwd")


# YAML field names recognised by ``--config``. Must match the snake-case
# :class:`SSOConfig` fields exposed on the argparse namespace. Listed
# explicitly (rather than inferred from ``SSOConfig.model_fields``) so that
# accidental additions to the model are surfaced as test failures rather
# than silently picked up.
_CONFIG_FIELDS = (
    "orography",
    "land_mask",
    "source_orography",
    "target_grid",
    "model_grid_type",
    "model_resolution",
    "effective_resolution",
    "output_grid",
    "output_dir",
    "grib_roundtrip",
    "dump_intermediates",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pproc-sso",
        description=(
            "Compute sub-grid scale orography (SSO) fields end-to-end. "
            "Reads a 5 km orography GRIB and a land-mask GRIB on the target "
            "grid, runs the ten-stage SSO pipeline, and writes the four "
            "output files (stdgwd, slogwd, anggwd, isogwd) to --output-dir."
        ),
    )

    # --- Required inputs (may also be supplied via --config YAML) -------
    # ``required=False`` so that a YAML config can satisfy them; we do
    # the required-field validation manually after merging the YAML in.
    parser.add_argument(
        "--orography",
        metavar="PATH",
        help="Path to source orography GRIB on the working grid.",
    )
    parser.add_argument(
        "--land-mask",
        dest="land_mask",
        metavar="PATH",
        help="Path to land-mask GRIB on the target grid.",
    )
    parser.add_argument(
        "--target-grid",
        dest="target_grid",
        metavar="GRID",
        help="Target/output grid spec, e.g. 'N256' or 'O1280'.",
    )

    # --- Optional inputs ------------------------------------------------
    parser.add_argument(
        "--source-orography",
        dest="source_orography",
        metavar="PATH",
        help=(
            "Optional fallback raw orography. Used by Stage 1 to (re)generate "
            "the working-grid orography when --orography does not exist."
        ),
    )
    parser.add_argument(
        "--model-grid-type",
        dest="model_grid_type",
        metavar="TYPE",
        help=(
            "Model grid family code ('O', 'N', 'F'). Auto-inferred from "
            "--target-grid when omitted."
        ),
    )
    parser.add_argument(
        "--model-resolution",
        dest="model_resolution",
        metavar="RES",
        type=int,
        help=(
            "Model nominal resolution (integer, e.g. 80). Auto-inferred from "
            "--target-grid when omitted."
        ),
    )
    parser.add_argument(
        "--effective-resolution",
        dest="effective_resolution",
        metavar="GRID",
        help=(
            "Override the auto-computed effective-resolution grid spec "
            "(e.g. 'N48'). Defaults to the value derived from the model grid."
        ),
    )
    parser.add_argument(
        "--output-grid",
        dest="output_grid",
        metavar="GRID",
        help="Output grid (OUT_RES) for working stages. Defaults to --target-grid.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        metavar="DIR",
        help="Directory in which to write the four output files (default: '.').",
    )

    # --- Behaviour toggles ---------------------------------------------
    parser.add_argument(
        "--grib-roundtrip",
        dest="grib_roundtrip",
        action="store_true",
        help=(
            "Encode/decode every numpy intermediate through GRIB to "
            "reproduce the per-step quantisation of the original ksh script. "
            "Narrows the value-array drift against the reference outputs."
        ),
    )
    parser.add_argument(
        "--dump-intermediates",
        dest="dump_intermediates",
        action="store_true",
        help=(
            "Write the 16 named intermediate GRIB files to --output-dir in "
            "addition to the four final outputs. Filenames are hard-coded; "
            "no user-controlled path component is involved."
        ),
    )

    # --- YAML config ----------------------------------------------------
    parser.add_argument(
        "--config",
        metavar="FILE",
        help=(
            "Optional YAML config file. Keys must match SSOConfig field "
            "names (snake_case). CLI arguments override YAML values."
        ),
    )

    return parser


def _load_yaml_config(path: str) -> dict:
    """Load and validate the YAML config at ``path``."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(
            f"--config: {path}: YAML root must be a mapping, got {type(data).__name__}"
        )
    unknown = set(data) - set(_CONFIG_FIELDS)
    if unknown:
        raise SystemExit(
            f"--config: {path}: unknown field(s): {sorted(unknown)!r}; "
            f"valid fields are {list(_CONFIG_FIELDS)!r}"
        )
    return data


def _merge_yaml_into_namespace(
    ns: argparse.Namespace, yaml_data: dict
) -> argparse.Namespace:
    """Apply YAML values to ``ns`` for any field the CLI left unset.

    For string/path/int fields the "unset" sentinel is ``None`` (argparse
    default for arguments without ``default=`` and without ``type=`` coercing
    a missing flag into a value). For boolean ``store_true`` flags the
    default is ``False``; YAML can flip them to ``True`` but cannot flip a
    CLI-passed ``True`` back to ``False`` (the user explicitly enabled it).
    """
    for key, value in yaml_data.items():
        current = getattr(ns, key, None)
        if isinstance(current, bool):
            # store_true default is False; only let YAML promote to True.
            if not current and bool(value):
                setattr(ns, key, True)
        elif current is None:
            setattr(ns, key, value)
    return ns


def _require(ns: argparse.Namespace, fields: Iterable[str]) -> None:
    """Raise ``SystemExit`` if any of ``fields`` is missing from ``ns``.

    argparse can't enforce ``required=True`` for fields that may legitimately
    come from YAML, so we validate after the YAML merge with the same
    error-channel argparse would use (sys.exit(2) via ``parser.error``-like
    semantics).
    """
    missing = [f for f in fields if getattr(ns, f, None) in (None, "")]
    if missing:
        flags = ", ".join("--" + f.replace("_", "-") for f in missing)
        raise SystemExit(
            f"pproc-sso: error: the following arguments are required "
            f"(via CLI or --config YAML): {flags}"
        )


def _build_config(ns: argparse.Namespace) -> SSOConfig:
    """Construct an :class:`SSOConfig` from the merged namespace."""
    kwargs: dict[str, Any] = {
        "orography": Path(ns.orography),
        "land_mask": Path(ns.land_mask),
        "target_grid": ns.target_grid,
    }
    if ns.source_orography is not None:
        kwargs["source_orography"] = Path(ns.source_orography)
    if ns.model_grid_type is not None:
        kwargs["model_grid_type"] = ns.model_grid_type
    if ns.model_resolution is not None:
        kwargs["model_resolution"] = int(ns.model_resolution)
    if ns.effective_resolution is not None:
        kwargs["effective_resolution"] = ns.effective_resolution
    if ns.output_grid is not None:
        kwargs["output_grid"] = ns.output_grid
    if ns.output_dir is not None:
        kwargs["output_dir"] = Path(ns.output_dir)
    kwargs["grib_roundtrip"] = bool(ns.grib_roundtrip)
    kwargs["dump_intermediates"] = bool(ns.dump_intermediates)
    return SSOConfig(**kwargs).resolve()


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    ns = parser.parse_args(argv)

    if ns.config:
        yaml_data = _load_yaml_config(ns.config)
        _merge_yaml_into_namespace(ns, yaml_data)

    _require(ns, ("orography", "land_mask", "target_grid"))

    config = _build_config(ns)

    results = compute_sso(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name in _OUTPUT_NAMES:
        (config.output_dir / name).write_bytes(results[name])


if __name__ == "__main__":  # pragma: no cover
    main()
