import pytest

from pproc.common.accumulation import (
    Difference,
    DifferenceRate,
    Mean,
    SimpleAccumulation,
)
from pproc.config.accumulation import LegacyStepAccumulation
from pproc.prob.window_manager import ThresholdWindowManager, AnomalyWindowManager


@pytest.mark.parametrize(
    "config, expected, exp_coords",
    [
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
            {
                f"{a}-{b}_0": (
                    SimpleAccumulation,
                    [{"comparison": "<=", "value": 273.15}],
                )
                for a, b in [(120, 240), (120, 168), (168, 240), (240, 360)]
            },
            set(range(126, 361, 6)),
            id="simple-range",
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
            {
                f"{a}-{b}_0": (
                    SimpleAccumulation,
                    [
                        {"comparison": ">=", "value": 15.0},
                        {"comparison": ">=", "value": 20.0},
                        {"comparison": ">=", "value": 25.0},
                    ],
                )
                for a, b in [(0, 24), (12, 36), (336, 360)]
            },
            set().union(range(6, 37, 6), range(342, 361, 6)),
            id="multi-range",
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
                    },
                ],
                "steps": [{"start_step": 6, "end_step": 360, "interval": 6}],
            },
            {
                **{
                    f"{a}-{b}_{i}": (
                        Difference,
                        [{"comparison": ">=", "value": thr} for thr in thrs],
                    )
                    for i, thrs in enumerate(
                        [[0.001, 0.005, 0.01, 0.02], [0.025, 0.05, 0.1]]
                    )
                    for a, b in [(0, 24), (12, 36), (336, 360)]
                },
                **{
                    f"{a}-{b}_2": (
                        DifferenceRate,
                        [
                            {"comparison": cmp, "value": val}
                            for cmp, vals in [("<", [0.001]), (">=", [0.003, 0.005])]
                            for val in vals
                        ],
                    )
                    for a, b in [(120, 240), (168, 240), (228, 360)]
                },
            },
            {12, 24, 36, 120, 168, 228, 240, 336, 360},
            id="diffs-range",
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
                    }
                ],
                "steps": [{"start_step": 0, "end_step": 360, "interval": 12}],
            },
            {
                f"{a}-{b}_0": (
                    Mean,
                    [
                        {"comparison": "<", "value": -2},
                        {"comparison": ">=", "value": 2},
                    ],
                )
                for a, b in [(120, 168), (168, 240), (240, 360)]
            },
            set(range(120, 361, 12)),
            id="mean-range",
        ),
    ],
)
def test_create_threshold(config, expected, exp_coords):
    win_mgr = ThresholdWindowManager({"step": LegacyStepAccumulation(**config)}, {})
    acc_mgr = win_mgr.mgr
    assert set(acc_mgr.accumulations.keys()) == set(expected.keys())
    assert set(win_mgr.window_thresholds.keys()) == set(expected.keys())
    for name in expected:
        accum = acc_mgr.accumulations[name]
        assert accum.name == name
        assert len(accum.dims) == 1
        assert accum.dims[0].key == "step"
        assert type(accum.dims[0].accumulation) == expected[name][0]
        assert win_mgr.window_thresholds[name] == expected[name][1]
    assert set(acc_mgr.coords.keys()) == {"step"}
    assert acc_mgr.coords["step"] == exp_coords


@pytest.mark.parametrize(
    "config, expected, exp_coords",
    [
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
                    }
                ],
                "steps": [{"start_step": 0, "end_step": 360, "interval": 12}],
            },
            {
                **{
                    f"{s}_{op}_0": (
                        SimpleAccumulation,
                        [{"comparison": cmp, "value": val} for val in vals],
                    )
                    for cmp, op, vals in [
                        ("<", "minimum", [-8, -4]),
                        (">", "maximum", [4, 8]),
                    ]
                    for s in [0, 12, 360]
                },
                **{
                    f"{a}-{b}_1": (
                        Mean,
                        [
                            {"comparison": "<", "value": -4},
                            {"comparison": ">=", "value": 2},
                        ],
                    )
                    for a, b in [(120, 240), (336, 360)]
                },
                **{
                    f"std_{s}_{op}_0": (
                        SimpleAccumulation,
                        [{"comparison": cmp, "value": val}],
                    )
                    for cmp, op, val in [
                        (">", "maximum", 1),
                        ("<", "minimum", -1.5),
                    ]
                    for s in [0, 12, 300]
                },
            },
            {0, 12, 300}.union(range(120, 241, 12), range(336, 361, 12)),
            id="multi",
        ),
    ],
)
def test_create_anomaly(config, expected, exp_coords):
    win_mgr = AnomalyWindowManager({"step": LegacyStepAccumulation(**config)}, {})
    acc_mgr = win_mgr.mgr
    assert set(acc_mgr.accumulations.keys()) == set(expected.keys())
    assert set(win_mgr.window_thresholds.keys()) == set(expected.keys())
    for name in expected:
        accum = acc_mgr.accumulations[name]
        assert accum.name == name
        assert len(accum.dims) == 1
        assert accum.dims[0].key == "step"
        assert type(accum.dims[0].accumulation) == expected[name][0]
        assert win_mgr.window_thresholds[name] == expected[name][1]
    assert set(acc_mgr.coords.keys()) == {"step"}
    assert acc_mgr.coords["step"] == exp_coords
