# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared config base for ``pproc-climate-fields`` products.

Conflator does NOT traverse discriminated-union branches when collecting
``CLIArg`` metadata, but it DOES traverse nested/base models correctly.
So the shared CLI options common to every product live on a single base
:class:`ConfigModel` (``BaseGenerateConfig``) that every product config
subclasses. Adding a new shared flag here surfaces it on every product's
``--help`` without further plumbing.

Design notes
------------
* ``target_grid`` is declared here as ``Optional[str]``: most products
  interpolate to a target grid, but a few (notably ``land-mask``) do
  not, so requiring it at the base would be surprising. Products that
  need it enforce presence via their own validator or by re-declaring
  the field as required.
* ``bits_per_value`` is declared here (with the ``--bits-per-value``
  flag) but is intentionally applied inside each product's
  ``generate()`` — the base cannot know which output metadata dicts
  to patch. Products that support it thread it through
  :func:`pproc.common.io.encode_grib`; products that ignore it simply
  don't read the field.
* ``grib_roundtrip`` is a debug knob used by the SSO pipeline; simple
  products (land-mask) ignore it.
"""

from __future__ import annotations

from typing import Annotated, Optional

from conflator import CLIArg, ConfigModel
from pydantic import Field

__all__ = ["BaseGenerateConfig"]


class BaseGenerateConfig(ConfigModel):
    """Base ``ConfigModel`` shared by every generate-climate-fields product.

    Every product subclasses this and adds its own input/output paths and
    product-specific flags. See
    :class:`pproc.climate.generate.products.land_mask.LandMaskConfig`
    for a small worked example.
    """

    target_grid: Annotated[
        Optional[str],
        CLIArg("--target-grid", default=None),
        Field(
            description=(
                "Target output grid spec (e.g. 'N256', 'O1280'). Maps to "
                "the operational $MIR_GTYPE_SET. Optional at this layer; "
                "products that need it require it via their own validator."
            ),
        ),
    ] = None

    verbose: Annotated[
        int,
        CLIArg("-v", "--verbose", action="count", default=None),
        Field(
            description=(
                "Increase logging verbosity to stdout: -v shows INFO, -vv "
                "shows DEBUG. Default: silent (WARNING)."
            ),
        ),
    ] = 0

    grib_roundtrip: Annotated[
        bool,
        CLIArg("--grib-roundtrip", action="store_true", default=None),
        Field(
            description=(
                "Encode/decode every numpy intermediate through GRIB to "
                "reproduce the per-step quantisation of the legacy ksh "
                "scripts. Honoured by products that support it (SSO); "
                "simple products ignore the flag."
            ),
        ),
    ] = False

    bits_per_value: Annotated[
        Optional[int],
        CLIArg("--bits-per-value", type=int, default=None),
        Field(
            gt=0,
            description=(
                "Optional GRIB bitsPerValue override applied at encode "
                "time. When None (default), eccodes' packing default "
                "applies (24 for grid_simple). Set to 32 to match the "
                "legacy ksh scripts' output precision."
            ),
        ),
    ] = None
