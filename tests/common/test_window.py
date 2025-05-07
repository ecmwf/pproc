import numpy as np
import pytest
from typing import Optional, Tuple, Union, Any

from pproc.common.accumulation import Accumulation, create_accumulation
from pproc.common.window import legacy_window_factory, translate_window_config


def create_window(
    coords: Union[list[Any], dict],
    window_operation: str,
    include_start: bool,
    grib_keys: Optional[dict] = None,
    deaccumulate: bool = False,
    return_name: bool = False,
    **extra,
) -> Union[Accumulation, Tuple[Accumulation, str]]:
    name, config = translate_window_config(
        coords, window_operation, include_start, grib_keys, deaccumulate, **extra
    )
    acc = create_accumulation(config)
    if return_name:
        return acc, name
    return acc


def test_instantaneous_window():
    accum = create_window([1], "aggregation", True)
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
        ["sum", [[3, 6, 9], [3, 6, 9]]],
    ],
)
def test_simple_op(window_operation, values):
    accum = create_window([0, 1, 2], window_operation, False)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    accum.feed(0, step_values)
    accum.feed(1, step_values)
    accum.feed(2, [[2, 4, 6], [1, 2, 3]])
    acc_values = accum.get_values()
    assert acc_values is not None
    np.testing.assert_equal(acc_values, values)


def test_multi_windows():
    accum = create_window([0, 1, 2], "sum", False)
    accum2 = create_window([0, 1, 2], "sum", False)
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
    "operation, include_init, steps, step_increment, values",
    [
        pytest.param("difference", True, [0, 2], 1, [[1, 2, 3], [2, 4, 6]], id="diff"),
        pytest.param(
            "weighted_mean",
            False,
            [0, 1, 2],
            1,
            [[1.5, 3, 4.5], [3, 6, 9]],
            id="weighted_mean",
        ),
        pytest.param(
            "difference_rate",
            True,
            [0, 240],
            120,
            np.divide([[1, 2, 3], [2, 4, 6]], 240),
            id="difference_rate",
        ),
        pytest.param(
            "mean", False, list(range(7)), 3, [[1.5, 3, 4.5], [3, 6, 9]], id="mean"
        ),
    ],
)
def test_windows(operation, include_init, steps, step_increment, values):
    accum = create_window(steps, operation, include_init)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    accum.feed(0, step_values)
    accum.feed(step_increment, step_values)
    accum.feed(2 * step_increment, step_values * 2)
    acc_values = accum.get_values()
    assert acc_values is not None
    np.testing.assert_almost_equal(acc_values, values)


@pytest.mark.parametrize(
    "steps, operation, extra_keys, grib_key_values",
    [
        pytest.param(
            [1], "aggregation", None, {"step": "1", "timeRangeIndicator": 0}, id="inst"
        ),
        pytest.param(
            [0],
            "aggregation",
            None,
            {"step": "0", "timeRangeIndicator": 1},
            id="inst-0",
        ),
        pytest.param(
            [260],
            "aggregation",
            None,
            {"step": "260", "timeRangeIndicator": 10},
            id="inst-260",
        ),
        pytest.param(
            [1, 2], "maximum", None, {"stepRange": "1-2", "stepType": "max"}, id="range"
        ),
        pytest.param(
            list(range(320, 361)),
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
def test_grib_header(steps, operation, extra_keys, grib_key_values):
    accum = create_window(steps, operation, True, extra_keys)
    header = accum.grib_keys()
    assert header == grib_key_values


@pytest.mark.parametrize(
    "config, grib_keys, expected",
    [
        pytest.param(
            {"windows": [{"coords": [[120], [123], [126], [129], [132], [360]]}]},
            {"mars.expver": "0001"},
            {
                f"{s}_0": {
                    "operation": "aggregation",
                    "coords": [s],
                    "sequential": True,
                    "metadata": {
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
            {"windows": [{"coords": [[0], [0, 3], [3, 6], [300, 306]]}]},
            {"timeRangeIndicator": 2},
            {
                "0_0": {
                    "operation": "aggregation",
                    "coords": [0],
                    "sequential": True,
                    "metadata": {"timeRangeIndicator": 2, "step": "0"},
                    "deaccumulate": False,
                },
                "0-3_0": {
                    "operation": "aggregation",
                    "coords": [3],
                    "sequential": True,
                    "metadata": {
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
                    "metadata": {
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
                    "metadata": {
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
                        "operation": "difference",
                        "coords": [
                            [a, b]
                            for a, b in [(90, 96), (93, 99), (96, 102), (270, 276)]
                        ],
                        "metadata": {"stepType": "diff", "bitsPerValue": 16},
                    },
                    {
                        "operation": "difference",
                        "coords": [
                            [a, b]
                            for a, b in [
                                (120, 144),
                                (240, 264),
                                (264, 288),
                                (240, 360),
                                (0, 360),
                            ]
                        ],
                        "metadata": {
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
                        "metadata": {
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
                        "metadata": {
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
                        "operation": "aggregation",
                        "coords": [[75], [78], [81], [84], [87], [90], [288]],
                        "metadata": {"bitsPerValue": 16},
                    }
                ]
            },
            {"expver": "0001"},
            {
                f"{s}_0": {
                    "operation": "aggregation",
                    "coords": [s],
                    "sequential": True,
                    "metadata": {
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
                        "operation": "aggregation",
                        "include_start": False,
                        "coords": [
                            list(range(12, 19, 6)),
                            list(range(330, 337, 6)),
                            list(range(336, 343, 6)),
                            list(range(342, 349, 6)),
                            list(range(348, 355, 6)),
                            list(range(354, 361, 6)),
                        ],
                        "metadata": {"bitsPerValue": 16},
                    }
                ]
            },
            {"expver": "0001"},
            {
                f"{a}-{b}_0": {
                    "operation": "aggregation",
                    "coords": [b],
                    "sequential": True,
                    "metadata": {
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
                        "operation": "mean",
                        "coords": [
                            list(range(12, 37, 6)),
                            list(range(60, 133, 6)),
                            list(range(0, 241, 6)),
                            list(range(0, 361, 24)),
                        ],
                        "metadata": {"timeRangeIndicator": 3},
                    }
                ]
            },
            {},
            {
                f"{a}-{b}_0": {
                    "operation": "mean",
                    "coords": list(range(a + s, b + 1, s)),
                    "sequential": True,
                    "metadata": {
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
                        "operation": "maximum",
                        "coords": [
                            {"from": 0, "to": 24, "by": 6},
                            {"from": 24, "to": 48, "by": 6},
                            {"from": 48, "to": 72, "by": 6},
                            {"from": 72, "to": 96, "by": 6},
                            {"from": 96, "to": 120, "by": 6},
                            {"from": 120, "to": 144, "by": 6},
                            {"from": 144, "to": 168, "by": 6},
                            {"from": 120, "to": 360, "by": 24},
                        ],
                        "metadata": {"timeRangeIndicator": 2},
                    }
                ]
            },
            {},
            {
                f"{a}-{b}_0": {
                    "operation": "maximum",
                    "coords": list(range(a + s, b + 1, s)),
                    "sequential": True,
                    "metadata": {
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
                        "operation": "minimum",
                        "coords": [
                            {"from": 0, "to": 24, "by": 6},
                            {"from": 24, "to": 48, "by": 6},
                            {"from": 48, "to": 72, "by": 6},
                            {"from": 72, "to": 96, "by": 6},
                            {"from": 96, "to": 120, "by": 6},
                            {"from": 120, "to": 144, "by": 6},
                            {"from": 144, "to": 168, "by": 6},
                            {"from": 120, "to": 360, "by": 24},
                        ],
                        "metadata": {"timeRangeIndicator": 2},
                    }
                ]
            },
            {},
            {
                f"{a}-{b}_0": {
                    "operation": "minimum",
                    "coords": list(range(a + s, b + 1, s)),
                    "sequential": True,
                    "metadata": {
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
                        "coords": [
                            {"from": a, "to": b, "by": c}
                            for a, b, c in [
                                (120, 240, 6),
                                (120, 168, 6),
                                (168, 240, 6),
                                (240, 360, 6),
                            ]
                        ],
                    }
                ],
            },
            {"type": "ep", "localDefinitionNumber": 5},
            {
                f"{a}-{b}_0": {
                    "operation": "minimum",
                    "coords": list(range(a + s, b + 1, s)),
                    "sequential": True,
                    "thresholds": [{"comparison": "<=", "value": 273.15}],
                    "metadata": {
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
                        "coords": [
                            {"from": a, "to": b, "by": c}
                            for a, b, c in [
                                (0, 24, 6),
                                (12, 36, 6),
                                (336, 360, 6),
                            ]
                        ],
                    }
                ],
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
                    "metadata": {
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
                        "operation": "difference",
                        "thresholds": [
                            {"comparison": ">=", "value": 0.001},
                            {"comparison": ">=", "value": 0.005},
                            {"comparison": ">=", "value": 0.01},
                            {"comparison": ">=", "value": 0.02},
                        ],
                        "coords": [
                            [a, b]
                            for a, b in [
                                (0, 24),
                                (12, 36),
                                (336, 360),
                            ]
                        ],
                        "metadata": {"stepType": "accum"},
                    },
                    {
                        "operation": "difference",
                        "thresholds": [
                            {"comparison": ">=", "value": 0.025},
                            {"comparison": ">=", "value": 0.05},
                            {"comparison": ">=", "value": 0.1},
                        ],
                        "coords": [
                            [a, b]
                            for a, b in [
                                (0, 24),
                                (12, 36),
                                (336, 360),
                            ]
                        ],
                        "metadata": {"stepType": "accum"},
                    },
                    {
                        "operation": "difference_rate",
                        "factor": 1.0 / 24.0,
                        "thresholds": [
                            {"comparison": "<", "value": 0.001},
                            {"comparison": ">=", "value": 0.003},
                            {"comparison": ">=", "value": 0.005},
                        ],
                        "coords": [
                            [a, b]
                            for a, b in [
                                (120, 240),
                                (168, 240),
                                (228, 360),
                            ]
                        ],
                        "metadata": {"stepType": "diff"},
                    },
                ],
            },
            {"type": "ep", "localDefinitionNumber": 5},
            {
                **{
                    f"{a}-{b}_{i}": {
                        "operation": "difference",
                        "coords": [a, b],
                        "sequential": True,
                        "thresholds": [
                            {"comparison": ">=", "value": thr} for thr in thrs
                        ],
                        "metadata": {
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
                        "metadata": {
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
                        "operation": "mean",
                        "include_start": True,
                        "thresholds": [
                            {"comparison": "<", "value": -2},
                            {"comparison": ">=", "value": 2},
                        ],
                        "coords": [
                            {"from": 120, "to": 168, "by": 12},
                            {"from": 168, "to": 240, "by": 12},
                            {"from": 240, "to": 360, "by": 12},
                        ],
                        "metadata": {"bitsPerValue": 24},
                    }
                ],
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
                    "metadata": {
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
                        "coords": [[0], [12], [360]],
                        "metadata": {"bitsPerValue": 24},
                    },
                    {
                        "operation": "mean",
                        "include_start": True,
                        "thresholds": [
                            {"comparison": "<", "value": -4},
                            {"comparison": ">=", "value": 2},
                        ],
                        "coords": [
                            {"from": 120, "to": 240, "by": 12},
                            {"from": 336, "to": 360, "by": 12},
                        ],
                        "metadata": {"bitsPerValue": 24},
                    },
                ],
                "std_anomaly_windows": [
                    {
                        "thresholds": [
                            {"comparison": ">", "value": 1},
                            {"comparison": "<", "value": -1.5},
                        ],
                        "coords": [[0], [12], [300]],
                        "metadata": {
                            "localDefinitionNumber": 30,
                            "bitsPerValue": 24,
                        },
                    }
                ],
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
                        "metadata": {
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
                        "metadata": {
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
                        "metadata": {
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
                        "operation": "mean",
                        "deaccumulate": True,
                        "coords": [[120], [123]],
                    }
                ]
            },
            {},
            {
                f"{s}_0": {
                    "operation": "mean",
                    "coords": [s],
                    "sequential": True,
                    "metadata": {
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
