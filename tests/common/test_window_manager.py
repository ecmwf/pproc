import pytest

from pproc.common.accumulation import Aggregation, Difference, Mean, SimpleAccumulation
from pproc.common.window_manager import WindowManager


@pytest.mark.parametrize(
    "config, expected, exp_coords",
    [
        pytest.param(
            {"windows": [{"coords": [[x] for x in [120, 123, 126, 129, 132, 360]]}]},
            {f"{s}_0": Aggregation for s in [120, 123, 126, 129, 132, 360]},
            {120, 123, 126, 129, 132, 360},
            id="simple",
        ),
        pytest.param(
            {
                "windows": [
                    {"coords": [[0], [0, 3], [3, 6], {"from": 300, "to": 306, "by": 6}]}
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
                        "operation": "difference",
                        "coords": [
                            [a, b]
                            for a, b in [(90, 96), (93, 99), (96, 102), (270, 276)]
                        ],
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
                        "operation": "none",
                        "coords": [[x] for x in [75, 78, 81, 84, 87, 90, 288]],
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
                        "operation": "none",
                        "include_start": False,
                        "coords": [
                            {"from": a, "to": b, "by": 6}
                            for a, b in [
                                (12, 18),
                                (330, 336),
                                (336, 342),
                                (342, 348),
                                (348, 354),
                                (354, 360),
                            ]
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
                        "operation": "mean",
                        "coords": [
                            {"from": a, "to": b, "by": 6}
                            for a, b in [(12, 36), (60, 132), (0, 240)]
                        ]
                        + [{"from": 0, "to": 360, "by": 24}],
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
                        "operation": "maximum",
                        "coords": [
                            {"from": a, "to": b, "by": 6}
                            for a, b in [
                                (0, 24),
                                (24, 48),
                                (48, 72),
                                (72, 96),
                                (96, 120),
                                (120, 144),
                                (144, 168),
                            ]
                        ]
                        + [{"from": 120, "to": 360, "by": 24}],
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
                        "operation": "minimum",
                        "coords": [
                            {"from": a, "to": b, "by": 6}
                            for a, b in [
                                (0, 24),
                                (24, 48),
                                (48, 72),
                                (72, 96),
                                (96, 120),
                                (120, 144),
                                (144, 168),
                            ]
                        ]
                        + [{"from": 120, "to": 360, "by": 24}],
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
        pytest.param(
            {
                "windows": [
                    {
                        "operation": "mean",
                        "include_start": True,
                        "coords": {
                            "type": "monthly",
                            "date": "20241001",
                            "from": 0,
                            "to": 5160,
                            "by": 6,
                        },
                    }
                ],
            },
            {
                f"{x}-{y}_0": Mean
                for x, y in [
                    (0, 744),
                    (744, 1464),
                    (1464, 2208),
                    (2208, 2952),
                    (2952, 3624),
                    (3624, 4368),
                    (4368, 5088),
                ]
            },
            {x for x in range(0, 5089, 6)},
            id="monthly",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "operation": "mean",
                        "coords": {
                            "type": "ranges",
                            "to": 360,
                            "width": 120,
                            "interval": 120,
                            "by": 6,
                        },
                    }
                ],
            },
            {
                f"{x}-{y}_0": Mean
                for x, y in [
                    (0, 120),
                    (120, 240),
                    (240, 360),
                ]
            },
            {x for x in range(6, 361, 6)},
            id="stepranges",
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
        "step": {
            "type": "legacywindow",
            "windows": [
                {
                    "operation": "mean",
                    "coords": [
                        {"from": 0, "to": 24, "by": 6},
                        {"from": 12, "to": 36, "by": 6},
                        {"from": 24, "to": 48, "by": 6},
                    ],
                }
            ],
        },
        "hdate": {
            "operation": "aggregation",
            "coords": [[20200218, 20210218, 20230218]],
        },
    }
    expected_accums = [
        "step_0-24_0:hdate_20200218-20230218",
        "step_12-36_0:hdate_20200218-20230218",
        "step_24-48_0:hdate_20200218-20230218",
    ]
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
