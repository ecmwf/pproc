# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``pproc-climate-fields`` console script.

Architecture
------------
Conflator has no native subcommand support, and its CLIArg walk stops at
discriminated-union boundaries — so it cannot expose the union of every
product's flags on a single parser. The proven pattern (matches
``pproc-clustereps``' multi-mode invocation) is:

1. A thin argparse layer strips the *field name* from ``argv`` and
   dispatches to the matching entry in
   :func:`pproc.climate.generate.registry.registry`.
2. That entry's :class:`~pproc.climate.generate.config.BaseGenerateConfig`
   subclass is handed to a per-field
   ``Conflator(app_name=f"pproc-climate-fields-{field}",
   model=ConfigCls)`` which parses the remaining args normally.

Conflator's ``load()`` reads ``sys.argv`` via
``parser.parse_known_args()`` — no argv is threaded through the API —
so the dispatcher rewrites ``sys.argv`` for the duration of the load
and restores it after. The ``prog`` component (``sys.argv[0]``) is set
to ``"pproc-climate-fields <field>"`` so the field's own
``--help`` output identifies itself correctly.

Verified: Conflator sees the field's own ``CLIArg`` flags. Test
coverage lives in ``pproc/tests/climate/generate/test_cli.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from typing import List, Optional

from conflator import Conflator

from pproc.climate._logging import configure_logging
from pproc.climate.generate.io import write_outputs
from pproc.climate.generate.registry import ProductEntry, registry

__all__ = ["main"]


logger = logging.getLogger("pproc.generate")


_PROG = "pproc-climate-fields"


def _format_field_listing(entries: dict[str, ProductEntry]) -> str:
    """Format the per-field listing shown by the dispatcher's ``--help``."""
    if not entries:
        return "  (no fields registered)"
    width = max(len(name) for name in entries)
    lines = []
    for name, entry in entries.items():
        lines.append(f"  {name:<{width}}  {entry.description}")
    return "\n".join(lines)


def _print_top_level_help(entries: dict[str, ProductEntry], stream) -> None:
    """Write the dispatcher's own ``--help`` output."""
    stream.write(
        f"usage: {_PROG} <field> [<field-flags>]\n\n"
        "Unified climate-field generation tool: pick a field, then pass "
        "the field-specific flags.\n\n"
        "Available fields:\n"
    )
    stream.write(_format_field_listing(entries))
    stream.write("\n\n")
    stream.write(f"Run '{_PROG} <field> --help' for a field's own flags.\n")


@contextlib.contextmanager
def _rewritten_argv(new_argv: List[str]):
    """Temporarily replace ``sys.argv`` and restore on exit.

    Conflator reads ``sys.argv`` via ``parser.parse_known_args()`` with
    no argv threading — this context manager is the least-invasive way
    to feed it a synthetic argv (the field-specific slice) and put the
    original argv back afterwards so tests running in the same process
    don't leak the mutation.
    """
    saved = sys.argv
    try:
        sys.argv = new_argv
        yield
    finally:
        sys.argv = saved


def main(argv: Optional[List[str]] = None) -> int:
    """Dispatch to a per-field product.

    Parameters
    ----------
    argv:
        Argument list *excluding* the program name (i.e. what
        ``sys.argv[1:]`` would give). ``None`` reads from
        ``sys.argv[1:]`` — the standard console-script convention.

    Returns
    -------
    int
        Process exit code (0 = success). Errors surface as
        ``SystemExit`` from argparse / Conflator / this function.
    """
    if argv is None:
        argv = sys.argv[1:]

    entries = registry()

    # Top-level help / no args → list fields.
    if not argv or argv[0] in ("-h", "--help"):
        _print_top_level_help(entries, sys.stdout)
        return 0

    field = argv[0]
    remaining = argv[1:]

    if field not in entries:
        # Match argparse-style: usage line to stderr, non-zero exit code.
        sys.stderr.write(f"{_PROG}: error: unknown field {field!r}\n\n")
        sys.stderr.write("Available fields:\n")
        sys.stderr.write(_format_field_listing(entries))
        sys.stderr.write("\n")
        return 2

    entry = entries[field]

    # Build a Conflator for the chosen product. app_name pattern gives
    # each field its own env-var namespace (PPROC_GENERATE_CLIMATE_FIELDS_SSO_*
    # etc.) and its own ~/.pproc-climate-fields-<field>.yaml
    # config file, without any of them clashing with each other or with
    # the dispatcher's own app_name.
    app_name = f"{_PROG}-{field}"
    prog = f"{_PROG} {field}"

    # Rewrite argv so Conflator's internal parse_known_args() reads only
    # this field's flags. Setting argv[0] to "prog <field>" keeps the
    # field's own --help output identifying itself as such rather than
    # the underlying app_name.
    synthetic_argv = [prog, *remaining]
    with _rewritten_argv(synthetic_argv):
        # Explicit argparser lets us pin ``prog`` so ``pproc-generate-
        # climate-fields <field> --help`` shows the right header rather
        # than the auto-generated one from Conflator's default parser.
        parser = argparse.ArgumentParser(prog=prog)
        try:
            cfg = Conflator(
                app_name=app_name, model=entry.config_cls, argparser=parser
            ).load()
        except SystemExit:
            # Preserve --help exit code (0) and validation exit code (2).
            raise

    # Logging comes AFTER config-load so we know the verbose count.
    configure_logging(cfg.verbose)

    logger.info("%s start field=%s", _PROG, field)

    try:
        results = entry.generate_fn(cfg)
    except (FileNotFoundError, ValueError) as exc:
        # Surface a clean non-zero exit (no Python traceback) for common
        # operator mistakes: missing input files, grid mismatches,
        # malformed grid specs. Preserves the ergonomics of the legacy
        # pproc-sso CLI, extended to every product.
        raise SystemExit(f"{prog}: error: {exc}") from exc

    write_outputs(results, cfg)

    logger.info("%s done field=%s", _PROG, field)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
