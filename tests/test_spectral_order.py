# coding: utf-8

from math import ceil

import pytest


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
        self.order = float(
            SpectralOrder._orders[order] if isinstance(order, str) else order
        )
        assert self.order

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


linear, quadratic, cubic = map(SpectralOrder, (1, 2, 3))

TESTS = (
    (linear, 15999, 8000),
    (linear, 7999, 4000),
    (linear, 3999, 2000),
    (linear, 3199, 1600),
    (linear, 2559, 1280),
    (linear, 2047, 1024),
    (linear, 1599, 800),
    (linear, 1279, 640),
    (linear, 1023, 512),
    (linear, 799, 400),
    (linear, 639, 320),
    (linear, 511, 256),
    (linear, 399, 200),
    (linear, 319, 160),
    (linear, 255, 128),
    (linear, 191, 96),
    (linear, 159, 80),
    (linear, 95, 48),
    (linear, 63, 32),
    (linear, 31, 16),
    (quadratic, 1706, 1280),
    # (quadratic, 1364, 1024)  # found in gaussgr, 1365 works
    (quadratic, 853, 640),
    (quadratic, 341, 256),
    (quadratic, 213, 160),
    (quadratic, 170, 128),
    (quadratic, 106, 80),
    (quadratic, 63, 48),
    (quadratic, 42, 32),
    (quadratic, 21, 16),
    (cubic, 7999, 8000),
    (cubic, 3999, 4000),
    (cubic, 2559, 2560),
    (cubic, 2047, 2048),
    (cubic, 1999, 2000),
    (cubic, 1599, 1600),
    (cubic, 1279, 1280),
    (cubic, 1023, 1024),
    (cubic, 911, 912),
    (cubic, 799, 800),
    (cubic, 639, 640),
    (cubic, 511, 512),
    (cubic, 399, 400),
    (cubic, 319, 320),
    (cubic, 255, 256),
    (cubic, 199, 200),
    (cubic, 191, 192),
    (cubic, 159, 160),
    (cubic, 95, 96),
    (cubic, 79, 80),
    (cubic, 63, 64),
)


@pytest.mark.parametrize("test", TESTS)
def test_spectral_order(test):
    so, t, n = test
    assert t == so.spectral_truncation(n)
    assert n == so.gaussian_number(t)


if __name__ == "__main__":
    for test in TESTS:
        try:
            test_spectral_order(test)
        except Exception as e:
            print(e)
