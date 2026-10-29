# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Field-name → (ConfigModel, generate_fn, description) registry.

Discovery is deliberately explicit: an imported list of product modules,
one entry per field. Auto-discovery via pkgutil would be marginally
tidier for the ~24 upcoming ports, but explicit keeps the dispatcher's
help output deterministic (registration order == help order) and makes
"add a new field" a one-line change in an obviously-named list.

Each product module MUST expose:

* ``FIELD_NAME: str``   — the CLI dispatch key (e.g. ``"land-mask"``).
* ``DESCRIPTION: str``  — the one-line ``--help`` line for the dispatcher.
* ``CONFIG: type[ConfigModel]`` — the Conflator config subclass.
* ``generate(config: ConfigModel) -> dict[str, bytes]`` — the algorithm.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from conflator import ConfigModel

from pproc.climate.generate.products import (
    albedo,
    albedo_four_stream,
    albedo_single_stream,
    aqua_planet,
    glacier_cover,
    glacier_mask,
    irrigation_cover,
    lake_cover,
    lake_depth,
    lake_mask,
    land_cover,
    land_mask,
    ocean_bathymetry,
    ocean_mask,
    oceanic_emissions,
    orography,
    orography_variance,
    sea_surface,
    soil_moisture,
    soil_moisture_smos,
    soil_type,
    soil_type_hwsd,
    sso,
    subgrid_orography_sdfor,
    urban_cover,
    water_type,
    wetland_cover,
)

__all__ = ["ProductEntry", "registry"]


class ProductEntry(NamedTuple):
    """One row of the registry."""

    config_cls: type[ConfigModel]
    generate_fn: Callable[[ConfigModel], dict[str, bytes]]
    description: str


# Ordered list of product modules; add new fields here as they are ported.
# Order controls the dispatcher's ``--help`` field-list order — kept
# alphabetical by FIELD_NAME.
_PRODUCTS = [
    albedo,
    albedo_four_stream,
    albedo_single_stream,
    aqua_planet,
    glacier_cover,
    glacier_mask,
    irrigation_cover,
    lake_cover,
    lake_depth,
    lake_mask,
    land_cover,
    land_mask,
    ocean_bathymetry,
    ocean_mask,
    oceanic_emissions,
    orography,
    orography_variance,
    sea_surface,
    soil_moisture,
    soil_moisture_smos,
    soil_type,
    soil_type_hwsd,
    sso,
    subgrid_orography_sdfor,
    urban_cover,
    water_type,
    wetland_cover,
]


def registry() -> dict[str, ProductEntry]:
    """Return the field-name → (config_cls, generate_fn, description) map.

    Built fresh on each call so tests can monkeypatch ``_PRODUCTS`` if
    they need a temporary mock product. Small enough that the rebuild
    cost is negligible next to Conflator's own initialisation.
    """
    entries: dict[str, ProductEntry] = {}
    for module in _PRODUCTS:
        name = module.FIELD_NAME
        if name in entries:
            raise RuntimeError(
                f"duplicate FIELD_NAME {name!r} in the product registry "
                f"({module.__name__} vs an earlier module)"
            )
        entries[name] = ProductEntry(
            config_cls=module.CONFIG,
            generate_fn=module.generate,
            description=module.DESCRIPTION,
        )
    return entries
