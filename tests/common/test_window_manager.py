import pytest

from pproc.common.accumulation import Aggregation, Difference, Mean, SimpleAccumulation
from pproc.common.window_manager import WindowManager


@pytest.mark.parametrize(
    "config, expected, exp_coords",
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
            {f"{s}_0": Aggregation for s in [120, 123, 126, 129, 132, 360]},
            {120, 123, 126, 129, 132, 360},
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
            {
                "0_0": Aggregation,
                "0-3_0": Aggregation,
                "3-6_0": Aggregation,
                "300-306_0": Aggregation,
            },
            {0, 3, 6, 306},
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
                    }
                ]
            },
            {
                f"{a}-{b}_0": Aggregation
                for a, b in [(0, 168), (120, 288), (528, 696), (936, 1104)]
            },
            {"0-168", "120-288", "528-696", "936-1104"},
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
                    },
                ]
            },
            {
                f"{a}-{b}_{i}": Difference
                for i, ranges in enumerate(
                    [
                        [(90, 96), (93, 99), (96, 102), (270, 276)],
                        [(120, 144), (240, 264), (264, 288), (240, 360), (0, 360)],
                    ]
                )
                for a, b in ranges
            },
            {0, 90, 93, 96, 99, 102, 120, 144, 240, 264, 270, 276, 288, 360},
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
                    }
                ],
                "steps": [{"start_step": 144, "end_step": 240, "interval": 6}],
            },
            {f"{a}-{b}_0": Difference for a, b in [(144, 150), (150, 156)]},
            {144, 150, 156},
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
                    }
                ]
            },
            {f"{s}_0": Aggregation for s in [75, 78, 81, 84, 87, 90, 288]},
            {75, 78, 81, 84, 87, 90, 288},
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
                    }
                ]
            },
            {
                f"{a}-{b}_0": Aggregation
                for a, b in [
                    (12, 18),
                    (330, 336),
                    (336, 342),
                    (342, 348),
                    (348, 354),
                    (354, 360),
                ]
            },
            {18, 336, 342, 348, 354, 360},
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
                    }
                ]
            },
            {f"{a}-{b}_0": Mean for a, b in [(12, 36), (60, 132), (0, 240), (0, 360)]},
            set().union(range(6, 241, 6), range(264, 361, 24)),
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
                    }
                ]
            },
            {
                f"{a}-{b}_0": SimpleAccumulation
                for a, b in [
                    (0, 24),
                    (24, 48),
                    (48, 72),
                    (72, 96),
                    (96, 120),
                    (120, 144),
                    (144, 168),
                    (120, 360),
                ]
            },
            set().union(range(6, 169, 6), range(144, 361, 24)),
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
                    }
                ]
            },
            {
                f"{a}-{b}_0": SimpleAccumulation
                for a, b in [
                    (0, 24),
                    (24, 48),
                    (48, 72),
                    (72, 96),
                    (96, 120),
                    (120, 144),
                    (144, 168),
                    (120, 360),
                ]
            },
            set().union(range(6, 169, 6), range(144, 361, 24)),
            id="min-range",
        ),
    ],
)
def test_create(config, expected, exp_coords):
    win_mgr = WindowManager(config, {})
    acc_mgr = win_mgr.mgr
    assert set(acc_mgr.accumulations.keys()) == set(expected.keys())
    for name in expected:
        accum = acc_mgr.accumulations[name]
        assert accum.name == name
        assert len(accum.dims) == 1
        assert accum.dims[0].key == "step"
        assert type(accum.dims[0].accumulation) == expected[name]
    assert set(acc_mgr.coords.keys()) == {"step"}
    assert acc_mgr.coords["step"] == exp_coords


def test_create_multidim():
    config = {
        "accumulations": {
            "step": {
                "type": "legacywindow",
                "windows": [
                    {
                        "window_operation": "mean",
                        "periods": [
                            {"range": [0, 24, 6]},
                            {"range": [12, 36, 6]},
                            {"range": [24, 48, 6]},
                        ],
                    }
                ],
            },
            "hdate": {
                "operation": "aggregation",
                "coords": [[20200218, 20210218, 20230218]],
            },
        }
    }
    expected_accums = ["step_0-24_0", "step_12-36_0", "step_24-48_0"]
    expected_dims = [("step", Mean), ("hdate", Aggregation)]
    expected_coords = {
        "step": set(range(6, 49, 6)),
        "hdate": {20200218, 20210218, 20230218},
    }

    win_mgr = WindowManager(config, {})
    acc_mgr = win_mgr.mgr
    assert set(acc_mgr.accumulations.keys()) == set(expected_accums)
    for name in expected_accums:
        accum = acc_mgr.accumulations[name]
        assert accum.name == name
        assert len(accum.dims) == len(expected_dims)
        for i, (dim, tp) in enumerate(expected_dims):
            assert accum.dims[i].key == dim
            assert type(accum.dims[i].accumulation) == tp
    assert set(acc_mgr.coords.keys()) == set(expected_coords.keys())
    for dim, exp_coords in expected_coords.items():
        assert acc_mgr.coords[dim] == exp_coords
