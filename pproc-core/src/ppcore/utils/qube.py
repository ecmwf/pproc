# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable

from qubed import Qube


def union(qubes: Iterable[Qube]) -> Qube:
    """
    Return the union of a list of Qubes
    """
    out = Qube.empty()
    for qube in qubes:
        out = out | qube
    return out


def qube_from_datacubes(datacubes: Iterable[dict]) -> Qube:
    """
    Return a Qube from a list of datacubes
    """
    out = Qube.empty()
    for cube in datacubes:
        out = out | Qube.from_datacube(cube)
    return out
