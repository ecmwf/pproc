# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unified climate-field generation framework.

Exposes the ``pproc-climate-fields`` console script (via
:mod:`pproc.climate.generate.__main__`) that dispatches to per-field
product modules under :mod:`pproc.climate.generate.products`. Each
product provides a Conflator ``ConfigModel`` (subclassing
:class:`~pproc.climate.generate.config.BaseGenerateConfig`) and a
``generate(config) -> dict[str, bytes]`` function that returns
logical-name → GRIB-bytes mappings. The CLI layer maps logical names
to user-supplied ``--<name>-out`` paths and writes the outputs; the
product never touches paths.

See :mod:`pproc.climate.generate.registry` for the field-to-product
registry and :mod:`pproc.climate.generate.__main__` for the argparse
dispatcher that hands off to a per-field Conflator.
"""

__all__: list[str] = []
