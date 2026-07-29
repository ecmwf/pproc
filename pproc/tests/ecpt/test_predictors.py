import datetime
import pytest
import numpy as np

from pproc.ecpt.predictors import _local_solar_time


@pytest.mark.parametrize(
    "hour, longitudes, expected",
    [
        [
            3,
            [-50, -10, 0, 10, 50],
            [23.66666667, 2.33333333, 3.0, 3.66666667, 6.33333333],
        ],
        [
            21,
            [-50, -10, 0, 10, 50],
            [17.6666667, 20.3333333, 21.0, 21.6666667, 0.3333333],
        ],
        [0, [-50, -10, 0, 10, 50], [20.6666667, 23.3333333, 0.0, 0.6666667, 3.3333333]],
    ],
)
def test_lst(hour, longitudes, expected):
    np.testing.assert_almost_equal(
        _local_solar_time(hour, np.asarray(longitudes)), np.asarray(expected)
    )
