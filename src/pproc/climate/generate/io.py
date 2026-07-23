# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CLI-layer output writer for ``pproc-climate-fields`` products.

Products return ``{logical_name: grib_bytes, ...}``; the CLI maps each
logical name to the user-supplied ``--<logical_name>-out`` path (a
``Path``-typed field ``<logical_name>_out`` on the config model) and
writes the bytes there. If a product returns a logical name with no
matching config field, that is a programming error and is raised loudly.

Templated outputs
-----------------
For products emitting many outputs that differ only by a numeric
discriminator (e.g. sea-surface's 12 monthly SST files), the framework
supports a template fallback: if a returned logical name of shape
``<prefix>_<digits>`` has no matching ``<prefix>_<digits>_out`` field,
the writer looks for ``<prefix>_out_template`` — a string field
containing a Python format-string placeholder ``{month}``. The trailing
digits are substituted (as ``int``) into the placeholder to produce
the concrete output path. This lets sea-surface's config carry a single
``sst_out_template`` field instead of 12 ``sst_MM_out`` Path fields.

The template convention deliberately supports only the ``{month}``
placeholder name (its intended use is monthly climatology outputs).
Other placeholder names or non-monthly discriminators are out of scope
for the current framework and would need an explicit design decision.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pproc.climate.generate.config import BaseGenerateConfig

__all__ = ["write_outputs"]


logger = logging.getLogger(__name__)


def _resolve_output_path(name: str, config: BaseGenerateConfig) -> Path:
    """Map a product's logical output name to its concrete filesystem path.

    Tries in order:

    1. ``<name>_out`` on ``config`` — the standard per-output flag path.
    2. ``<prefix>_out_template`` where ``<prefix>_<digits>`` is the shape
       of ``name`` — the template fallback for numeric-discriminator
       multi-output products. Substitutes the digits as ``{month}``
       (converted to ``int`` so that ``{month:02d}``-style padding works).

    If neither field exists on the config, raises ``AttributeError``
    with a message pinpointing what was expected. This is a programming
    error in the product (its returned dict key doesn't match either
    contract) and is meant to be caught early during development.
    """
    per_output_attr = f"{name}_out"
    if hasattr(config, per_output_attr):
        return _as_path(getattr(config, per_output_attr))

    # Template fallback: recognise names of shape "<prefix>_<digits>".
    if "_" in name:
        prefix, _, suffix = name.rpartition("_")
        if suffix.isdigit():
            template_attr = f"{prefix}_out_template"
            if hasattr(config, template_attr):
                template = getattr(config, template_attr)
                return _as_path(template.format(month=int(suffix)))

    raise AttributeError(
        f"product returned logical output {name!r} but config "
        f"{type(config).__name__} has neither field {per_output_attr!r} "
        f"(expected a Path with CLIArg '--{name.replace('_', '-')}-out') "
        f"nor a matching '<prefix>_out_template' field. This is a "
        f"programming error in the product."
    )


def _as_path(value: object) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def write_outputs(
    results: dict[str, bytes],
    config: BaseGenerateConfig,
) -> None:
    """Write each ``results[name]`` payload to its configured output path.

    Two contracts are supported for locating the output path — see the
    module docstring and :func:`_resolve_output_path`. Products MUST NOT
    build filenames themselves; this function is the one place where
    logical names meet the filesystem.

    Parameters
    ----------
    results:
        Mapping of logical output name to the encoded GRIB bytes.
    config:
        The resolved product config; must expose ``<name>_out`` (or a
        ``<prefix>_out_template``) for every ``name`` in ``results``.

    Raises
    ------
    AttributeError
        If ``config`` exposes neither ``<name>_out`` nor a matching
        template field for some logical output the product returned.
    """
    for name, payload in results.items():
        out_path = _resolve_output_path(name, config)
        # Parent directory: create on demand so the ksh wrappers don't
        # have to. Matches the behaviour of the legacy pproc-sso CLI.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(payload)
        logger.info("wrote %s → %s (%d bytes)", name, out_path, len(payload))
