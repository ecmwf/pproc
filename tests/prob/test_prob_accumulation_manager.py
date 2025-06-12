# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from pproc.common.accumulation import (
    Difference,
    DifferenceRate,
    Mean,
    SimpleAccumulation,
)
from pproc.prob.accumulation_manager import (
    ThresholdAccumulationManager,
    AnomalyAccumulationManager,
)


@pytest.mark.parametrize(
    "config, expected, exp_coords",
    [
        pytest.param(
            {
                "type": "legacywindow",
                "windows": [
                    {
                        "thresholds": [
                            {"comparison": "<=", "value": 273.15},
                        ],
                        "coords": [
                            {"from": 120, "to": 240, "by": 6},
                            {"from": 120, "to": 168, "by": 6},
                            {"from": 168, "to": 240, "by": 6},
                            {"from": 240, "to": 360, "by": 6},
                        ],
                    }
                ],
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
                "type": "legacywindow",
                "windows": [
                    {
                        "thresholds": [
                            {"comparison": ">=", "value": 15},
                            {"comparison": ">=", "value": 20},
                            {"comparison": ">=", "value": 25},
                        ],
                        "coords": [
                            {"from": 0, "to": 24, "by": 6},
                            {"from": 12, "to": 36, "by": 6},
                            {"from": 336, "to": 360, "by": 6},
                        ],
                    }
                ],
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
                "type": "legacywindow",
                "windows": [
                    {
                        "operation": "difference",
                        "thresholds": [
                            {"comparison": ">=", "value": 0.001},
                            {"comparison": ">=", "value": 0.005},
                            {"comparison": ">=", "value": 0.01},
                            {"comparison": ">=", "value": 0.02},
                        ],
                        "coords": [[0, 24], [12, 36], [336, 360]],
                    },
                    {
                        "operation": "difference",
                        "thresholds": [
                            {"comparison": ">=", "value": 0.025},
                            {"comparison": ">=", "value": 0.05},
                            {"comparison": ">=", "value": 0.1},
                        ],
                        "coords": [[0, 24], [12, 36], [336, 360]],
                    },
                    {
                        "operation": "difference_rate",
                        "factor": 1.0 / 24.0,
                        "thresholds": [
                            {"comparison": "<", "value": 0.001},
                            {"comparison": ">=", "value": 0.003},
                            {"comparison": ">=", "value": 0.005},
                        ],
                        "coords": [[120, 240], [168, 240], [228, 360]],
                    },
                ],
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
            {0, 12, 24, 36, 120, 168, 228, 240, 336, 360},
            id="diffs-range",
        ),
        pytest.param(
            {
                "type": "legacywindow",
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
                    }
                ],
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
    acc_mgr = ThresholdAccumulationManager.create({"step": config}, {})
    assert set(acc_mgr.accumulations.keys()) == set(expected.keys())
    assert set(acc_mgr._thresholds.keys()) == set(expected.keys())
    for name in expected:
        accum = acc_mgr.accumulations[name]
        assert accum.name == name
        assert len(accum.dims) == 1
        assert accum.dims[0].key == "step"
        assert type(accum.dims[0].accumulation) == expected[name][0]
        assert acc_mgr._thresholds[name] == expected[name][1]
    assert set(acc_mgr.coords.keys()) == {"step"}
    assert acc_mgr.coords["step"] == exp_coords


@pytest.mark.parametrize(
    "config, expected, exp_coords",
    [
        pytest.param(
            {
                "type": "legacywindow",
                "windows": [
                    {
                        "thresholds": [
                            {"comparison": "<", "value": -8},
                            {"comparison": "<", "value": -4},
                            {"comparison": ">", "value": 4},
                            {"comparison": ">", "value": 8},
                        ],
                        "coords": [[0], [12], [360]],
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
                    },
                ],
                "std_anomaly_windows": [
                    {
                        "thresholds": [
                            {"comparison": ">", "value": 1},
                            {"comparison": "<", "value": -1.5},
                        ],
                        "coords": [[0], [12], [300]],
                    }
                ],
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
    acc_mgr = AnomalyAccumulationManager.create({"step": config}, {})
    assert set(acc_mgr.accumulations.keys()) == set(expected.keys())
    assert set(acc_mgr._thresholds.keys()) == set(expected.keys())
    for name in expected:
        accum = acc_mgr.accumulations[name]
        assert accum.name == name
        assert len(accum.dims) == 1
        assert accum.dims[0].key == "step"
        assert type(accum.dims[0].accumulation) == expected[name][0]
        assert acc_mgr._thresholds[name] == expected[name][1]
    assert set(acc_mgr.coords.keys()) == {"step"}
    assert acc_mgr.coords["step"] == exp_coords
