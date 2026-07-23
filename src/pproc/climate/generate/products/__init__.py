# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-field product modules for ``pproc-climate-fields``.

Every product exposes:

* ``FIELD_NAME`` — the CLI field name (e.g. ``"land-mask"``).
* ``CONFIG`` — the Conflator ``ConfigModel`` subclass (of
  :class:`~pproc.climate.generate.config.BaseGenerateConfig`) that
  captures its input/output flags.
* ``generate(config) -> dict[str, bytes]`` — the algorithm, returning
  one entry per logical output name. Metadata (shortName, packingType,
  paramId, bitsPerValue, ...) is applied inside ``generate()`` via
  :func:`pproc.common.io.encode_grib`; the CLI layer never touches it.
* ``DESCRIPTION`` — a one-line human-readable description used by the
  dispatcher's ``--help`` listing.

Products must not build output filenames themselves; they only return
logical-name → bytes mappings. See
:mod:`pproc.climate.generate.io.write_outputs` for the CLI layer that
maps logical names to ``--<name>-out`` paths.
"""

__all__: list[str] = []
