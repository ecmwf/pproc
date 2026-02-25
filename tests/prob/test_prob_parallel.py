import pytest
import numpy as np

from pproc.prob.threshold import ThresholdConfig
from pproc.prob.parallel import ensemble_probability


@pytest.mark.parametrize(
    "array, config, expected",
    [
        [
            [[0, 0, 1], [0, 1, 1]],
            {"limit_type": "lower", "value": 0.5, "comparison": ">"},
            [0, 50, 100],
        ],
        [
            [[0, 0, 1], [0, 1, 1], [1, 2, 3], [3, 1, 4]],
            {
                "param_thresholds": [
                    {"value": 0.5, "comparison": ">"},
                    {"value": 2, "comparison": "<"},
                ]
            },
            [0, 50, 0],
        ],
        [
            [[0, 0.4, 0.8], [0.4, 0.8, 1.2]],
            {
                "type": "range",
                "lower_value": 0.5,
                "lower_comparison": ">",
                "upper_value": 1,
                "upper_comparison": "<",
            },
            [0, 50, 50],
        ],
    ],
    ids=["single-thr", "multi-param", "multi-thr"],
)
def test_ensemble_probability(array, config, expected):
    threshold_config = ThresholdConfig(
        out_paramid=1,
        **config,
    )

    probabilities = ensemble_probability(np.asarray(array), threshold_config)
    assert probabilities.shape == (3,)
    np.testing.assert_array_equal(
        expected, ensemble_probability(np.asarray(array), threshold_config)
    )
