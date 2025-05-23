# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# coding: utf-8

from math import ceil


class SpectralOrder:
    _orders = dict(
        l_2=1,  # linear, gaussian=reduced
        l_full=1,  # linear, gaussian=regular
        _2=2,  # quadratic, gaussian=reduced
        _full=2,  # quadratic, gaussian=regular
        _3=3,  # cubic, non-octahedral grid
        _4=3,  # cubic, octahedral grid
    )

    def __init__(self, order):
        self.order = float(SpectralOrder._orders.get(order, order))
        assert self.order > 0

    def spectral_truncation(self, n: float):
        assert n > 0
        t = ceil(4 * n / (self.order + 1)) - 1
        assert t > 0
        return t

    def gaussian_number(self, t: float):
        assert t > 0
        n = int((t + 1) * (self.order + 1) / 4)
        assert n > 0
        return n
