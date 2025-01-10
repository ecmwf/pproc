import numpy as np
import pytest

from pproc.common.window import (
    create_window,
    legacy_window_factory,
    parse_window_config,
)


@pytest.mark.parametrize(
    "steps, include_init, exp",
    [
        pytest.param([0, 0], True, [0], id="0-0"),
        pytest.param([3, 10], True, list(range(3, 11)), id="3-10"),
        pytest.param([2, 6], False, list(range(3, 7)), id="2-6-noinit"),
        pytest.param([3, 12, 3], True, [3, 6, 9, 12], id="3-12-by3"),
        pytest.param([1, 1, 6], True, [1], id="1-1-by6"),
        pytest.param([0, 1, 4], True, [0], id="0-1-by4"),
        pytest.param([0, 24, 8], False, [8, 16, 24], id="0-24-by8-noinit"),
    ],
)
def test_window_steps(steps, include_init, exp):
    window = parse_window_config({"range": steps}, include_init)
    assert window.steps == exp


def test_instantaneous_window():
    accum = create_window({"range": [1, 1]}, "none", True)
    step_values = np.array([[1, 1, 1], [2, 2, 2]])
    accum.feed(0, step_values)
    assert accum.get_values() is None
    accum.feed(1, step_values)
    values = accum.get_values()
    assert values is not None
    np.testing.assert_equal(values, step_values)


@pytest.mark.parametrize(
    "window_operation, values",
    [
        ["minimum", [[1, 2, 3], [1, 2, 3]]],
        ["maximum", [[2, 4, 6], [2, 4, 6]]],
        ["add", [[3, 6, 9], [3, 6, 9]]],
    ],
)
def test_simple_op(window_operation, values):
    accum = create_window({"range": [0, 2]}, window_operation, False)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    accum.feed(0, step_values)
    accum.feed(1, step_values)
    accum.feed(2, [[2, 4, 6], [1, 2, 3]])
    acc_values = accum.get_values()
    assert acc_values is not None
    np.testing.assert_equal(acc_values, values)


def test_multi_windows():
    accum = create_window({"range": [0, 2]}, "add", False)
    accum2 = create_window({"range": [0, 2]}, "add", False)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    accum.feed(1, step_values)
    accum2.feed(1, step_values)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    accum.feed(2, step_values * 2)
    values = accum.get_values()
    assert values is not None
    np.testing.assert_equal(values, np.array([[3, 6, 9], [6, 12, 18]]))
    accum2.feed(2, step_values)
    values2 = accum2.get_values()
    assert values2 is not None
    np.testing.assert_equal(values2, np.array([[2, 4, 6], [4, 8, 12]]))


@pytest.mark.parametrize(
    "operation, include_init, end_step, step_increment, values",
    [
        pytest.param("diff", True, 2, 1, [[1, 2, 3], [2, 4, 6]], id="diff"),
        pytest.param(
            "weightedsum", False, 2, 1, [[1.5, 3, 4.5], [3, 6, 9]], id="weightedsum"
        ),
        pytest.param(
            "diffdailyrate",
            True,
            240,
            120,
            np.divide([[1, 2, 3], [2, 4, 6]], 10),
            id="diffdailyrate",
        ),
        pytest.param("mean", False, 6, 3, [[1.5, 3, 4.5], [3, 6, 9]], id="mean"),
    ],
)
def test_windows(operation, include_init, end_step, step_increment, values):
    accum = create_window({"range": [0, end_step]}, operation, include_init)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    accum.feed(0, step_values)
    accum.feed(step_increment, step_values)
    accum.feed(2 * step_increment, step_values * 2)
    acc_values = accum.get_values()
    assert acc_values is not None
    np.testing.assert_almost_equal(acc_values, values)


@pytest.mark.parametrize(
    "start_end, operation, extra_keys, grib_key_values",
    [
        pytest.param(
            [1, 1], "none", None, {"step": "1", "timeRangeIndicator": 0}, id="inst"
        ),
        pytest.param(
            [0, 0], "none", None, {"step": "0", "timeRangeIndicator": 1}, id="inst-0"
        ),
        pytest.param(
            [260, 260],
            "none",
            None,
            {"step": "260", "timeRangeIndicator": 10},
            id="inst-260",
        ),
        pytest.param(
            [1, 2], "maximum", None, {"stepRange": "1-2", "stepType": "max"}, id="range"
        ),
        pytest.param(
            [320, 360],
            "maximum",
            None,
            {"stepRange": "320-360", "stepType": "max", "unitOfTimeRange": 11},
            id="range-360",
        ),
        pytest.param(
            [1, 2],
            "mean",
            {"stepType": "avg", "numberIncludedInAverage": 2},
            {"numberIncludedInAverage": 2, "stepRange": "1-2", "stepType": "avg"},
            id="extra",
        ),
    ],
)
def test_grib_header(start_end, operation, extra_keys, grib_key_values):
    accum = create_window({"range": start_end}, operation, True, extra_keys)
    header = accum.grib_keys()
    assert header == grib_key_values


@pytest.mark.parametrize(
    "config, grib_keys, expected",
    [
        pytest.param(
            {
                "windows": [
                    {
                        "periods": [
                            {"range": [120, 120]},
                            {"range": [123, 123]},
                            {"range": [126, 126]},
                            {"range": [129, 129]},
                            {"range": [132, 132]},
                            {"range": [360, 360]},
                        ]
                    }
                ]
            },
            {"mars.expver": "0001"},
            {
                f"{s}_0": {
                    "operation": "aggregation",
                    "coords": [s],
                    "sequential": True,
                    "grib_keys": {
                        "mars.expver": "0001",
                        "timeRangeIndicator": (0 if s < 256 else 10),
                        "step": str(s),
                    },
                    "deaccumulate": False,
                }
                for s in [120, 123, 126, 129, 132, 360]
            },
            id="simple",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "periods": [
                            {"range": [0, 0, 3]},
                            {"range": [0, 3, 3]},
                            {"range": [3, 6, 3]},
                            {"range": [300, 306, 6]},
                        ]
                    }
                ]
            },
            {"timeRangeIndicator": 2},
            {
                "0_0": {
                    "operation": "aggregation",
                    "coords": [0],
                    "sequential": True,
                    "grib_keys": {"timeRangeIndicator": 2, "step": "0"},
                    "deaccumulate": False,
                },
                "0-3_0": {
                    "operation": "aggregation",
                    "coords": [3],
                    "sequential": True,
                    "grib_keys": {
                        "timeRangeIndicator": 2,
                        "stepRange": "0-3",
                        "stepType": "max",
                    },
                    "deaccumulate": False,
                },
                "3-6_0": {
                    "operation": "aggregation",
                    "coords": [6],
                    "sequential": True,
                    "grib_keys": {
                        "timeRangeIndicator": 2,
                        "stepRange": "3-6",
                        "stepType": "max",
                    },
                    "deaccumulate": False,
                },
                "300-306_0": {
                    "operation": "aggregation",
                    "coords": [306],
                    "sequential": True,
                    "grib_keys": {
                        "unitOfTimeRange": 11,
                        "timeRangeIndicator": 2,
                        "stepRange": "300-306",
                        "stepType": "max",
                    },
                    "deaccumulate": False,
                },
            },
            id="simple-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "window_operation": "precomputed",
                        "periods": [
                            {"range": [0, 168]},
                            {"range": [120, 288]},
                            {"range": [528, 696]},
                            {"range": [936, 1104]},
                        ],
                        "grib_set": {"timeRangeIndicator": 3},
                    }
                ]
            },
            {},
            {
                f"{a}-{b}_0": {
                    "operation": "aggregation",
                    "coords": [f"{a}-{b}"],
                    "sequential": True,
                    "grib_keys": {
                        "timeRangeIndicator": 3,
                        "stepRange": f"{a}-{b}",
                        "stepType": "max",
                        **({} if b < 256 else {"unitOfTimeRange": 11}),
                    },
                    "deaccumulate": False,
                }
                for a, b in [
                    (0, 168),
                    (120, 288),
                    (528, 696),
                    (936, 1104),
                ]
            },
            id="precomputed",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "window_operation": "diff",
                        "periods": [
                            {"range": [90, 96]},
                            {"range": [93, 99]},
                            {"range": [96, 102]},
                            {"range": [270, 276]},
                        ],
                        "grib_set": {"stepType": "diff", "bitsPerValue": 16},
                    },
                    {
                        "window_operation": "diff",
                        "periods": [
                            {"range": [120, 144, 24]},
                            {"range": [240, 264, 24]},
                            {"range": [264, 288, 24]},
                            {"range": [240, 360, 120]},
                            {"range": [0, 360, 360]},
                        ],
                        "grib_set": {
                            "timeRangeIndicator": 5,
                            "gribTablesVersionNo": 132,
                        },
                    },
                ]
            },
            {},
            {
                **{
                    f"{a}-{b}_0": {
                        "operation": "difference",
                        "coords": [a, b],
                        "sequential": True,
                        "grib_keys": {
                            "stepType": "diff",
                            "bitsPerValue": 16,
                            "stepRange": f"{a}-{b}",
                            **({} if b < 256 else {"unitOfTimeRange": 11}),
                        },
                        "deaccumulate": False,
                    }
                    for a, b in [(90, 96), (93, 99), (96, 102), (270, 276)]
                },
                **{
                    f"{a}-{b}_1": {
                        "operation": "difference",
                        "coords": [a, b],
                        "sequential": True,
                        "grib_keys": {
                            "stepType": "max",
                            "timeRangeIndicator": 5,
                            "gribTablesVersionNo": 132,
                            "stepRange": f"{a}-{b}",
                            **({} if b < 256 else {"unitOfTimeRange": 11}),
                        },
                        "deaccumulate": False,
                    }
                    for a, b in [
                        (120, 144),
                        (240, 264),
                        (264, 288),
                        (240, 360),
                        (0, 360),
                    ]
                },
            },
            id="diff",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "window_operation": "diff",
                        "periods": [
                            {"range": [144, 150]},
                            {"range": [150, 156]},
                        ],
                        "grib_set": {"stepType": "diff", "bitsPerValue": 16},
                    }
                ],
                "steps": [{"start_step": 144, "end_step": 240, "interval": 6}],
            },
            {"expver": "0001"},
            {
                f"{a}-{b}_0": {
                    "operation": "difference",
                    "coords": [a, b],
                    "sequential": True,
                    "grib_keys": {
                        "stepType": "diff",
                        "bitsPerValue": 16,
                        "expver": "0001",
                        "stepRange": f"{a}-{b}",
                    },
                    "deaccumulate": False,
                }
                for a, b in [(144, 150), (150, 156)]
            },
            id="diff-steps",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "window_operation": "none",
                        "periods": [
                            {"range": [75, 75]},
                            {"range": [78, 78]},
                            {"range": [81, 81]},
                            {"range": [84, 84]},
                            {"range": [87, 87]},
                            {"range": [90, 90]},
                            {"range": [288, 288]},
                        ],
                        "grib_set": {"bitsPerValue": 16},
                    }
                ]
            },
            {"expver": "0001"},
            {
                f"{s}_0": {
                    "operation": "aggregation",
                    "coords": [s],
                    "sequential": True,
                    "grib_keys": {
                        "bitsPerValue": 16,
                        "expver": "0001",
                        "step": str(s),
                        "timeRangeIndicator": (0 if s < 256 else 10),
                    },
                    "deaccumulate": False,
                }
                for s in [75, 78, 81, 84, 87, 90, 288]
            },
            id="noop-simple",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "window_operation": "none",
                        "include_start_step": False,
                        "periods": [
                            {"range": [12, 18, 6]},
                            {"range": [330, 336, 6]},
                            {"range": [336, 342, 6]},
                            {"range": [342, 348, 6]},
                            {"range": [348, 354, 6]},
                            {"range": [354, 360, 6]},
                        ],
                        "grib_set": {"bitsPerValue": 16},
                    }
                ]
            },
            {"expver": "0001"},
            {
                f"{a}-{b}_0": {
                    "operation": "aggregation",
                    "coords": [b],
                    "sequential": True,
                    "grib_keys": {
                        "stepType": "max",
                        "bitsPerValue": 16,
                        "expver": "0001",
                        "stepRange": f"{a}-{b}",
                        **({} if b < 256 else {"unitOfTimeRange": 11}),
                    },
                    "deaccumulate": False,
                }
                for a, b in [
                    (12, 18),
                    (330, 336),
                    (336, 342),
                    (342, 348),
                    (348, 354),
                    (354, 360),
                ]
            },
            id="noop-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "window_operation": "mean",
                        "periods": [
                            {"range": [12, 36, 6]},
                            {"range": [60, 132, 6]},
                            {"range": [0, 240, 6]},
                            {"range": [0, 360, 24]},
                        ],
                        "grib_set": {"timeRangeIndicator": 3},
                    }
                ]
            },
            {},
            {
                f"{a}-{b}_0": {
                    "operation": "mean",
                    "coords": list(range(a + s, b + 1, s)),
                    "sequential": True,
                    "grib_keys": {
                        "timeRangeIndicator": 3,
                        "stepType": "max",
                        "stepRange": f"{a}-{b}",
                        **({} if b < 256 else {"unitOfTimeRange": 11}),
                    },
                    "deaccumulate": False,
                }
                for a, b, s in [
                    (12, 36, 6),
                    (60, 132, 6),
                    (0, 240, 6),
                    (0, 360, 24),
                ]
            },
            id="mean-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "window_operation": "maximum",
                        "periods": [
                            {"range": [0, 24, 6]},
                            {"range": [24, 48, 6]},
                            {"range": [48, 72, 6]},
                            {"range": [72, 96, 6]},
                            {"range": [96, 120, 6]},
                            {"range": [120, 144, 6]},
                            {"range": [144, 168, 6]},
                            {"range": [120, 360, 24]},
                        ],
                        "grib_set": {"timeRangeIndicator": 2},
                    }
                ]
            },
            {},
            {
                f"{a}-{b}_0": {
                    "operation": "maximum",
                    "coords": list(range(a + s, b + 1, s)),
                    "sequential": True,
                    "grib_keys": {
                        "timeRangeIndicator": 2,
                        "stepType": "max",
                        "stepRange": f"{a}-{b}",
                        **({} if b < 256 else {"unitOfTimeRange": 11}),
                    },
                    "deaccumulate": False,
                }
                for a, b, s in [
                    (0, 24, 6),
                    (24, 48, 6),
                    (48, 72, 6),
                    (72, 96, 6),
                    (96, 120, 6),
                    (120, 144, 6),
                    (144, 168, 6),
                    (120, 360, 24),
                ]
            },
            id="max-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "window_operation": "minimum",
                        "periods": [
                            {"range": [0, 24, 6]},
                            {"range": [24, 48, 6]},
                            {"range": [48, 72, 6]},
                            {"range": [72, 96, 6]},
                            {"range": [96, 120, 6]},
                            {"range": [120, 144, 6]},
                            {"range": [144, 168, 6]},
                            {"range": [120, 360, 24]},
                        ],
                        "grib_set": {"timeRangeIndicator": 2},
                    }
                ]
            },
            {},
            {
                f"{a}-{b}_0": {
                    "operation": "minimum",
                    "coords": list(range(a + s, b + 1, s)),
                    "sequential": True,
                    "grib_keys": {
                        "timeRangeIndicator": 2,
                        "stepType": "max",
                        "stepRange": f"{a}-{b}",
                        **({} if b < 256 else {"unitOfTimeRange": 11}),
                    },
                    "deaccumulate": False,
                }
                for a, b, s in [
                    (0, 24, 6),
                    (24, 48, 6),
                    (48, 72, 6),
                    (72, 96, 6),
                    (96, 120, 6),
                    (120, 144, 6),
                    (144, 168, 6),
                    (120, 360, 24),
                ]
            },
            id="min-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "thresholds": [
                            {"comparison": "<=", "value": 273.15},
                        ],
                        "periods": [
                            {"range": [120, 240]},
                            {"range": [120, 168]},
                            {"range": [168, 240]},
                            {"range": [240, 360]},
                        ],
                    }
                ],
                "steps": [{"start_step": 126, "end_step": 360, "interval": 6}],
            },
            {"type": "ep", "localDefinitionNumber": 5},
            {
                f"{a}-{b}_0": {
                    "operation": "minimum",
                    "coords": list(range(a + s, b + 1, s)),
                    "sequential": True,
                    "thresholds": [{"comparison": "<=", "value": 273.15}],
                    "grib_keys": {
                        "type": "ep",
                        "localDefinitionNumber": 5,
                        "stepType": "max",
                        "stepRange": f"{a}-{b}",
                        **({} if b < 256 else {"unitOfTimeRange": 11}),
                    },
                    "deaccumulate": False,
                }
                for a, b, s in [
                    (120, 240, 6),
                    (120, 168, 6),
                    (168, 240, 6),
                    (240, 360, 6),
                ]
            },
            id="simple-threshold-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "thresholds": [
                            {"comparison": ">=", "value": 15},
                            {"comparison": ">=", "value": 20},
                            {"comparison": ">=", "value": 25},
                        ],
                        "periods": [
                            {"range": [0, 24]},
                            {"range": [12, 36]},
                            {"range": [336, 360]},
                        ],
                    }
                ],
                "steps": [{"start_step": 6, "end_step": 360, "interval": 6}],
            },
            {"type": "ep", "localDefinitionNumber": 5},
            {
                f"{a}-{b}_0": {
                    "operation": "maximum",
                    "coords": list(range(a + s, b + 1, s)),
                    "sequential": True,
                    "thresholds": [
                        {"comparison": ">=", "value": 15.0},
                        {"comparison": ">=", "value": 20.0},
                        {"comparison": ">=", "value": 25.0},
                    ],
                    "grib_keys": {
                        "type": "ep",
                        "localDefinitionNumber": 5,
                        "stepType": "max",
                        "stepRange": f"{a}-{b}",
                        **({} if b < 256 else {"unitOfTimeRange": 11}),
                    },
                    "deaccumulate": False,
                }
                for a, b, s in [
                    (0, 24, 6),
                    (12, 36, 6),
                    (336, 360, 6),
                ]
            },
            id="multi-threshold-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "window_operation": "diff",
                        "thresholds": [
                            {"comparison": ">=", "value": 0.001},
                            {"comparison": ">=", "value": 0.005},
                            {"comparison": ">=", "value": 0.01},
                            {"comparison": ">=", "value": 0.02},
                        ],
                        "periods": [
                            {"range": [0, 24]},
                            {"range": [12, 36]},
                            {"range": [336, 360]},
                        ],
                        "grib_set": {"stepType": "accum"},
                    },
                    {
                        "window_operation": "diff",
                        "thresholds": [
                            {"comparison": ">=", "value": 0.025},
                            {"comparison": ">=", "value": 0.05},
                            {"comparison": ">=", "value": 0.1},
                        ],
                        "periods": [
                            {"range": [0, 24]},
                            {"range": [12, 36]},
                            {"range": [336, 360]},
                        ],
                        "grib_set": {"stepType": "accum"},
                    },
                    {
                        "window_operation": "diffdailyrate",
                        "thresholds": [
                            {"comparison": "<", "value": 0.001},
                            {"comparison": ">=", "value": 0.003},
                            {"comparison": ">=", "value": 0.005},
                        ],
                        "periods": [
                            {"range": [120, 240]},
                            {"range": [168, 240]},
                            {"range": [228, 360]},
                        ],
                        "grib_set": {"stepType": "diff"},
                    },
                ],
                "steps": [{"start_step": 6, "end_step": 360, "interval": 6}],
            },
            {"type": "ep", "localDefinitionNumber": 5},
            {
                **{
                    f"{a}-{b}_{i}": {
                        "operation": "difference",
                        "coords": [a, b] if a >= 6 else [b],
                        "sequential": True,
                        "thresholds": [
                            {"comparison": ">=", "value": thr} for thr in thrs
                        ],
                        "grib_keys": {
                            "type": "ep",
                            "localDefinitionNumber": 5,
                            "stepType": "accum",
                            "stepRange": f"{a}-{b}",
                            **({} if b < 256 else {"unitOfTimeRange": 11}),
                        },
                        "deaccumulate": False,
                    }
                    for i, thrs in enumerate(
                        [[0.001, 0.005, 0.01, 0.02], [0.025, 0.05, 0.1]]
                    )
                    for a, b in [
                        (0, 24),
                        (12, 36),
                        (336, 360),
                    ]
                },
                **{
                    f"{a}-{b}_2": {
                        "operation": "difference_rate",
                        "factor": 1.0 / 24.0,
                        "coords": [a, b],
                        "sequential": True,
                        "thresholds": [
                            {"comparison": cmp, "value": val}
                            for cmp, vals in [("<", [0.001]), (">=", [0.003, 0.005])]
                            for val in vals
                        ],
                        "grib_keys": {
                            "type": "ep",
                            "localDefinitionNumber": 5,
                            "stepType": "diff",
                            "stepRange": f"{a}-{b}",
                            **({} if b < 256 else {"unitOfTimeRange": 11}),
                        },
                        "deaccumulate": False,
                    }
                    for a, b in [
                        (120, 240),
                        (168, 240),
                        (228, 360),
                    ]
                },
            },
            id="diffs-threshold-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "window_operation": "mean",
                        "include_start_step": True,
                        "thresholds": [
                            {"comparison": "<", "value": -2},
                            {"comparison": ">=", "value": 2},
                        ],
                        "periods": [
                            {"range": [120, 168]},
                            {"range": [168, 240]},
                            {"range": [240, 360]},
                        ],
                        "grib_set": {"bitsPerValue": 24},
                    }
                ],
                "steps": [{"start_step": 0, "end_step": 360, "interval": 12}],
            },
            {"type": "ep", "localDefinitionNumber": 5, "bitsPerValue": 8},
            {
                f"{a}-{b}_0": {
                    "operation": "mean",
                    "coords": list(range(a, b + 1, s)),
                    "sequential": True,
                    "thresholds": [
                        {"comparison": "<", "value": -2},
                        {"comparison": ">=", "value": 2},
                    ],
                    "grib_keys": {
                        "type": "ep",
                        "localDefinitionNumber": 5,
                        "bitsPerValue": 24,
                        "stepType": "max",
                        "stepRange": f"{a}-{b}",
                        **({} if b < 256 else {"unitOfTimeRange": 11}),
                    },
                    "deaccumulate": False,
                }
                for a, b, s in [
                    (120, 168, 12),
                    (168, 240, 12),
                    (240, 360, 12),
                ]
            },
            id="mean-threshold-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "thresholds": [
                            {"comparison": "<", "value": -8},
                            {"comparison": "<", "value": -4},
                            {"comparison": ">", "value": 4},
                            {"comparison": ">", "value": 8},
                        ],
                        "periods": [
                            {"range": [0, 0]},
                            {"range": [12, 12]},
                            {"range": [360, 360]},
                        ],
                        "grib_set": {"bitsPerValue": 24},
                    },
                    {
                        "window_operation": "mean",
                        "include_start_step": True,
                        "thresholds": [
                            {"comparison": "<", "value": -4},
                            {"comparison": ">=", "value": 2},
                        ],
                        "periods": [
                            {"range": [120, 240]},
                            {"range": [336, 360]},
                        ],
                        "grib_set": {"bitsPerValue": 24},
                    },
                ],
                "std_anomaly_windows": [
                    {
                        "thresholds": [
                            {"comparison": ">", "value": 1},
                            {"comparison": "<", "value": -1.5},
                        ],
                        "periods": [
                            {"range": [0, 0]},
                            {"range": [12, 12]},
                            {"range": [300, 300]},
                        ],
                        "grib_set": {
                            "localDefinitionNumber": 30,
                            "bitsPerValue": 24,
                        },
                    }
                ],
                "steps": [{"start_step": 0, "end_step": 360, "interval": 12}],
            },
            {"type": "ep", "localDefinitionNumber": 5, "bitsPerValue": 8},
            {
                **{
                    f"{s}_{op}_0": {
                        "operation": op,
                        "coords": [s],
                        "sequential": True,
                        "thresholds": [
                            {"comparison": cmp, "value": val} for val in vals
                        ],
                        "grib_keys": {
                            "type": "ep",
                            "localDefinitionNumber": 5,
                            "bitsPerValue": 24,
                            "step": str(s),
                            "timeRangeIndicator": tri,
                        },
                        "deaccumulate": False,
                    }
                    for cmp, op, vals in [
                        ("<", "minimum", [-8, -4]),
                        (">", "maximum", [4, 8]),
                    ]
                    for s, tri in [(0, 1), (12, 0), (360, 10)]
                },
                **{
                    f"{a}-{b}_1": {
                        "operation": "mean",
                        "coords": list(range(a, b + 1, s)),
                        "sequential": True,
                        "thresholds": [
                            {"comparison": "<", "value": -4},
                            {"comparison": ">=", "value": 2},
                        ],
                        "grib_keys": {
                            "type": "ep",
                            "localDefinitionNumber": 5,
                            "bitsPerValue": 24,
                            "stepType": "max",
                            "stepRange": f"{a}-{b}",
                            **({} if b < 256 else {"unitOfTimeRange": 11}),
                        },
                        "deaccumulate": False,
                    }
                    for a, b, s in [
                        (120, 240, 12),
                        (336, 360, 12),
                    ]
                },
                **{
                    f"std_{s}_{op}_0": {
                        "operation": op,
                        "coords": [s],
                        "sequential": True,
                        "thresholds": [{"comparison": cmp, "value": val}],
                        "grib_keys": {
                            "type": "ep",
                            "localDefinitionNumber": 30,
                            "bitsPerValue": 24,
                            "step": str(s),
                            "timeRangeIndicator": tri,
                        },
                        "deaccumulate": False,
                    }
                    for cmp, op, val in [
                        (">", "maximum", 1),
                        ("<", "minimum", -1.5),
                    ]
                    for s, tri in [(0, 1), (12, 0), (300, 10)]
                },
            },
            id="multi-anomaly",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "window_operation": "mean",
                        "deaccumulate": True,
                        "periods": [
                            {"range": [120, 120]},
                            {"range": [123, 123]},
                        ],
                    }
                ]
            },
            {},
            {
                f"{s}_0": {
                    "operation": "mean",
                    "coords": [s],
                    "sequential": True,
                    "grib_keys": {
                        "timeRangeIndicator": 0,
                        "step": str(s),
                    },
                    "deaccumulate": True,
                }
                for s in [120, 123]
            },
            id="deaccumulation",
        ),
    ],
)
def test_legacy_window_factory(config, grib_keys, expected):
    configs = dict(legacy_window_factory(config, grib_keys))
    assert configs == expected
