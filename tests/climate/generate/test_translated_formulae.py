# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the mir-compute ``=`` → pproc-formula ``==`` translations.

Every place in the ported climate products where an equality comparison was
rewritten (see the per-product docstrings) is exercised here on a tiny
synthetic input to prove the rewritten formula evaluates as intended.
"""

from __future__ import annotations

import numpy as np

from pproc.formula import evaluate_formula


def test_soil_type_missing_replacement() -> None:
    """``field - (field == 9999) * 9998`` maps 9999 to 1, else unchanged."""
    field = np.array([1.0, 2.0, 9999.0, 5.0, 9999.0])
    result = evaluate_formula("field - (field == 9999) * 9998", {"field": field})
    np.testing.assert_allclose(result, [1.0, 2.0, 1.0, 5.0, 1.0])


def test_albedo_zero_over_land_gets_fill() -> None:
    """``field * land_mask + 0.15 * (field == 0) * land_mask``: zero-on-land → 0.15."""
    field = np.array([0.0, 0.3, 0.0, 0.5])
    land = np.array([1.0, 1.0, 0.0, 1.0])
    result = evaluate_formula(
        "field * land_mask + 0.15 * (field == 0) * land_mask",
        {"field": field, "land_mask": land},
    )
    # index 0: land, field=0 → 0.15
    # index 1: land, field=0.3 → 0.3
    # index 2: ocean, field=0 → 0 (land_mask is 0)
    # index 3: land, field=0.5 → 0.5
    np.testing.assert_allclose(result, [0.15, 0.3, 0.0, 0.5])


def test_albedo_exact_0149_over_glacier_gets_08() -> None:
    """``0.8 * (0.149 == field) * glacier_mask``: exact-0.149 on glacier → 0.8."""
    field = np.array([0.149, 0.15, 0.149])
    glacier = np.array([1.0, 1.0, 0.0])
    result = evaluate_formula(
        "0.8 * (0.149 == field) * glacier_mask",
        {"field": field, "glacier_mask": glacier},
    )
    np.testing.assert_allclose(result, [0.8, 0.0, 0.0])


def test_orography_variance_law_of_total_variance() -> None:
    """``abs(mean_square - mean * mean)`` returns non-negative variance."""
    # Contrived: variance of {1, 2, 3} is 2/3; mean=2; mean_square = (1+4+9)/3 = 14/3
    result = evaluate_formula(
        "abs( mean_square - (mean * mean) )",
        {"mean_square": np.array([14.0 / 3.0]), "mean": np.array([2.0])},
    )
    np.testing.assert_allclose(result, [2.0 / 3.0])


def test_oceanic_emissions_masking_formula() -> None:
    """``field * ocean_mask - 99 * (1 - ocean_mask)`` masks non-ocean to -99."""
    field = np.array([1.5, 2.0, 3.0])
    ocean = np.array([1.0, 0.0, 1.0])
    result = evaluate_formula(
        "field * ocean_mask - 99 * (1 - ocean_mask)",
        {"field": field, "ocean_mask": ocean},
    )
    np.testing.assert_allclose(result, [1.5, -99.0, 3.0])


def test_sea_surface_ocean_masking_9999() -> None:
    """``field * ocean_mask + (1 - ocean_mask) * 9999`` masks non-ocean to 9999."""
    field = np.array([280.0, 285.0])
    ocean = np.array([1.0, 0.0])
    result = evaluate_formula(
        "field * (ocean_mask) + (1 - ocean_mask) * 9999",
        {"field": field, "ocean_mask": ocean},
    )
    np.testing.assert_allclose(result, [280.0, 9999.0])


def test_soil_type_hwsd_class_5_extreme_clay() -> None:
    """``5 * (clay > 60)``: clay > 60% forces class 5."""
    clay = np.array([50.0, 65.0, 61.0, 60.0])
    result = evaluate_formula("5 * (clay > 60)", {"clay": clay})
    np.testing.assert_allclose(result, [0.0, 5.0, 5.0, 0.0])


def test_soil_type_hwsd_oceanic_carbon_override_type_6() -> None:
    """Composite override of low soil types to 6 when oceanic-carbon > 10."""
    result = evaluate_formula(
        "6 * (soil_types <= 2) * (oceanic_carbon > 10) + "
        "soil_types * (soil_types > 2) * (oceanic_carbon > 10) + "
        "soil_types * (oceanic_carbon <= 10)",
        {
            "soil_types": np.array([1.0, 3.0, 2.0]),
            "oceanic_carbon": np.array([15.0, 20.0, 5.0]),
        },
    )
    # index 0: type<=2 && oc>10 → 6
    # index 1: type>2 && oc>10 → 3 (preserved)
    # index 2: oc<=10 → 2 (preserved)
    np.testing.assert_allclose(result, [6.0, 3.0, 2.0])


def test_glacier_mask_composition() -> None:
    """glacier_mask = threshold(cicecap) * land_mask, then glacier_free = land - glacier."""
    cicecap = np.array([0.4, 0.6, 0.7, 0.9])
    land = np.array([1.0, 1.0, 0.0, 1.0])
    glacier = evaluate_formula(
        "(glacier_cover > 0.5) * 1 + 0", {"glacier_cover": cicecap}
    )
    glacier = evaluate_formula(
        "glacier_mask * land_mask",
        {"glacier_mask": glacier, "land_mask": land},
    )
    glacier_free = evaluate_formula(
        "land_mask - glacier_mask",
        {"land_mask": land, "glacier_mask": glacier},
    )
    # cicecap>0.5: [F, T, T, T]; & land [T, T, F, T] → glacier [0, 1, 0, 1]
    # glacier_free = land - glacier = [1, 0, 0, 0]
    np.testing.assert_allclose(glacier, [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(glacier_free, [1.0, 0.0, 0.0, 0.0])
