# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Registry-level smoke tests for the 27 pproc-climate-fields products.

Every product must:

* Register a unique ``FIELD_NAME`` in the dispatcher's registry.
* Expose a ``CONFIG`` class with an alphabetically-orderable ``FIELD_NAME``.
* Support ``pproc-climate-fields <field> --help`` — i.e., its ``CONFIG``
  parses successfully with just ``--help`` and exits 0.

These tests catch registration typos, import errors, and gross CLI setup
regressions across the full product surface without depending on any GRIB
inputs.
"""

from __future__ import annotations

import pytest

from pproc.climate.generate.registry import registry


EXPECTED_FIELDS = {
    "albedo",
    "albedo-four-stream",
    "albedo-single-stream",
    "aqua-planet",
    "glacier-cover",
    "glacier-mask",
    "irrigation-cover",
    "lake-cover",
    "lake-depth",
    "lake-mask",
    "land-cover",
    "land-mask",
    "ocean-bathymetry",
    "ocean-mask",
    "oceanic-emissions",
    "orography",
    "orography-variance",
    "sea-surface",
    "soil-moisture",
    "soil-moisture-smos",
    "soil-type",
    "soil-type-hwsd",
    "sso",
    "subgrid-orography-sdfor",
    "urban-cover",
    "water-type",
    "wetland-cover",
}


def test_all_27_products_registered() -> None:
    """Every field the task lists must appear in the runtime registry."""
    found = set(registry())
    assert found == EXPECTED_FIELDS, (
        f"missing: {EXPECTED_FIELDS - found}, extra: {found - EXPECTED_FIELDS}"
    )


def test_registry_entries_expose_the_contract() -> None:
    """Every entry must give us (config_cls, generate_fn, description)."""
    for name, entry in registry().items():
        assert entry.config_cls is not None, name
        assert callable(entry.generate_fn), name
        assert isinstance(entry.description, str) and entry.description, name


@pytest.mark.parametrize("field", sorted(EXPECTED_FIELDS))
def test_field_help_exit_zero(field: str) -> None:
    """``pproc-climate-fields <field> --help`` must exit 0 without ImportError."""
    from pproc.climate.generate.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main([field, "--help"])
    assert exc.value.code == 0
