# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Internal logging helper shared by the climate-fields CLIs.

Exposes a single :func:`configure_logging` function that maps an argparse
``-v`` count to a root-logger configuration, with output streamed to
**stdout** (not stderr — operator preference). Two formats are used:

* INFO    → ``[<logger>] <message>``
* DEBUG   → ``<HH:MM:SS> [<logger>] <message>``

The leading underscore marks the module as a ``pproc``-internal API; the
public ``pproc.climate`` surface remains focused on the SSO / field-calc /
gradient operators.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["configure_logging"]


# Sentinel attribute on the handler so :func:`configure_logging` can find and
# replace the handler it installed previously without disturbing handlers an
# embedding application may have added on its own.
_HANDLER_TAG = "_pproc_climate_cli_handler"


def configure_logging(verbose: int) -> None:
    """Configure root logging based on the CLI ``-v`` count.

    Parameters
    ----------
    verbose:
        Count produced by argparse's ``action='count', default=0`` for the
        ``-v`` / ``--verbose`` flag. ``0`` keeps the root logger silent
        (WARNING), ``1`` emits INFO, ``>=2`` emits DEBUG.

    Notes
    -----
    Output is streamed to :data:`sys.stdout`. The function is **idempotent**
    within a process: repeated calls reconfigure level/format on the same
    handler instance rather than accumulating duplicates. Handlers installed
    by other code (e.g. application-level configuration) are left in place.
    """
    if verbose <= 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    if level == logging.DEBUG:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(fmt="[%(name)s] %(message)s")

    root = logging.getLogger()
    # Replace any handler we previously installed; leave foreign handlers
    # alone so an embedding application keeps its own logging surface.
    existing = next(
        (h for h in root.handlers if getattr(h, _HANDLER_TAG, False)),
        None,
    )
    if existing is not None:
        # Reuse the handler: rebind stream (in case sys.stdout was swapped
        # by a test harness or pytest's capture machinery) and refresh
        # level + formatter to match the new verbosity request.
        #
        # We assign ``handler.stream`` directly rather than calling
        # ``handler.setStream``: the latter flushes the *current* stream
        # before swapping, which raises ``ValueError`` if pytest's
        # captured-stdout buffer has been closed between calls (a
        # frequent occurrence when multiple CLI invocations share a
        # process).
        existing.stream = sys.stdout
        existing.setLevel(level)
        existing.setFormatter(formatter)
    else:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)

    root.setLevel(level)
