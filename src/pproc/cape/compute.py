# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compute CAPE and CIN from pressure-level and surface data.

Wraps :func:`earthkit.meteo.thermo.cape_cin` with configurable
parcel type (``"mu"`` for most-unstable, ``"ml"`` for mixed-layer)
and optional layer depth.  Until the function is available in
earthkit-meteo, a dummy implementation is used for testing.
"""

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


def _dummy_cape_cin(
    p: ArrayLike,
    zh: ArrayLike,
    t: ArrayLike,
    q: ArrayLike,
    p_sfc: ArrayLike,
    zh_sfc: ArrayLike,
    t_sfc: ArrayLike,
    q_sfc: ArrayLike,
    parcel_type: str,
    layer_depth: float | None = None,
    extra_outputs: list | None = None,
    vertical_axis: int = 0,
    ept_method: str = "bolton43",
    lcl_method: str = "davies",
) -> Tuple[ArrayLike, ArrayLike]:
    """Dummy cape_cin returning zero arrays.  Used for testing until
    :func:`earthkit.meteo.thermo.cape_cin` is released."""
    horiz_shape = p.shape[1:] if vertical_axis == 0 else p.shape[:-1]
    cape = np.zeros(horiz_shape, dtype=np.float64)
    cin = np.zeros(horiz_shape, dtype=np.float64)
    return cape, cin


def _get_cape_cin_func():
    """Return the real cape_cin if available, else the dummy."""
    try:
        from earthkit.meteo.thermo import cape_cin
        return cape_cin
    except ImportError:
        return _dummy_cape_cin


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
    layer_depth: float | None = None,
) -> Tuple[ArrayLike, ArrayLike]:
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
    layer_depth : float or None
        Depth of the layer in Pa (e.g. 5000.0 for 50 hPa).
        Used by ``"mixed"`` and ``"mu"`` parcel types.

    Returns
    -------
    cape : array-like
        CAPE (J/kg), shape ``(n_points,)``.
    cin : array-like
        CIN (J/kg), shape ``(n_points,)``.
    """
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    zh = np.asarray(zh, dtype=np.float64)
    t_sfc = np.asarray(t_sfc, dtype=np.float64)
    q_sfc = np.asarray(q_sfc, dtype=np.float64)
    zh_sfc = np.asarray(zh_sfc, dtype=np.float64)
    p_sfc = np.asarray(p_sfc, dtype=np.float64)

    n_levels = t.shape[0]
    n_points = t.shape[1] if t.ndim > 1 else 1

    # Build pressure array: broadcast level list (hPa → Pa) to field shape
    p_pa = np.array(p_levels_hpa, dtype=np.float64) * 100.0
    p = np.broadcast_to(p_pa[:, np.newaxis], (n_levels, n_points))

    cape_cin_func = _get_cape_cin_func()

    kwargs = dict(
        p=p,
        zh=zh,
        t=t,
        q=q,
        p_sfc=p_sfc,
        zh_sfc=zh_sfc,
        t_sfc=t_sfc,
        q_sfc=q_sfc,
        parcel_type=parcel_type,
        vertical_axis=0,
    )
    if layer_depth is not None:
        kwargs["layer_depth"] = layer_depth

    cape, cin = cape_cin_func(**kwargs)
    return cape, cin
