# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compute CAPE and CIN from pressure-level and surface data.

Dispatches to the earthkit-meteo CAPE/CIN functions according to the
parcel type: ``"surface"`` (:func:`earthkit.meteo.thermo.surface_cape_cin`),
``"mixed"`` (:func:`earthkit.meteo.thermo.mixed_layer_cape_cin`) or
``"mu"`` (:func:`earthkit.meteo.thermo.most_unstable_cape_cin`).
"""

from typing import Any, Callable, Tuple

import numpy as np
from earthkit.meteo.thermo import (
    mixed_layer_cape_cin,
    most_unstable_cape_cin,
    surface_cape_cin,
)
from numpy.typing import ArrayLike

CAPE_CIN_FUNCTIONS: dict[str, Callable[..., Tuple[np.ndarray, np.ndarray]]] = {
    "surface": surface_cape_cin,
    "mixed": mixed_layer_cape_cin,
    "mu": most_unstable_cape_cin,
}


def compute_cape_cin(
    t: ArrayLike,
    q: ArrayLike,
    zh: ArrayLike,
    p_levels_hpa: list[int],
    t_sfc: ArrayLike,
    q_sfc: ArrayLike,
    zh_sfc: ArrayLike,
    p_sfc: ArrayLike,
    parcel_type: str = "mu",
    **options: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute CAPE and CIN on pressure levels.

    Parameters
    ----------
    t : array-like
        Temperature on pressure levels (K), shape ``(n_levels, n_points)``.
        Levels must be ordered by ascending pressure (TOA first).
    q : array-like
        Specific humidity on pressure levels (kg/kg), same shape as ``t``.
    zh : array-like
        Geopotential height on pressure levels (m), same shape as ``t``.
    p_levels_hpa : list of int
        Pressure levels in hPa, ascending order (e.g. [50, 100, …, 1000]).
    t_sfc : array-like
        Surface temperature (K), shape ``(n_points,)``.
    q_sfc : array-like
        Surface specific humidity (kg/kg), shape ``(n_points,)``.
    zh_sfc : array-like
        Surface geopotential height (m), shape ``(n_points,)``.
    p_sfc : array-like
        Surface pressure (Pa), shape ``(n_points,)``.
    parcel_type : str
        Parcel type for the CAPE/CIN computation.  ``"mu"`` for
        most-unstable, ``"mixed"`` for mixed-layer, ``"surface"``
        for surface parcel.
    **options
        Parcel-specific keyword arguments forwarded to the earthkit-meteo
        function, e.g. ``layer_depth`` (Pa) for ``"mixed"``, or
        ``exclude_surface_layer`` and ``max_search_height`` (m) for ``"mu"``.

    Returns
    -------
    cape : np.ndarray
        CAPE (J/kg), shape ``(n_points,)``.
    cin : np.ndarray
        CIN (J/kg), shape ``(n_points,)``.

    Raises
    ------
    ValueError
        If ``parcel_type`` is not supported.
    """
    if parcel_type not in CAPE_CIN_FUNCTIONS:
        raise ValueError(
            f"Unsupported parcel_type '{parcel_type}', "
            f"expected one of {list(CAPE_CIN_FUNCTIONS)}"
        )

    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    zh = np.asarray(zh, dtype=np.float64)
    t_sfc = np.asarray(t_sfc, dtype=np.float64)
    q_sfc = np.asarray(q_sfc, dtype=np.float64)
    zh_sfc = np.asarray(zh_sfc, dtype=np.float64)
    p_sfc = np.asarray(p_sfc, dtype=np.float64)

    # Broadcast level list (hPa → Pa) to the profile shape
    p_pa = np.array(p_levels_hpa, dtype=np.float64) * 100.0
    p = np.broadcast_to(p_pa.reshape((-1,) + (1,) * (t.ndim - 1)), t.shape)

    return CAPE_CIN_FUNCTIONS[parcel_type](
        p, t, q, zh, p_sfc, t_sfc, q_sfc, zh_sfc, vertical_axis=0, **options
    )
