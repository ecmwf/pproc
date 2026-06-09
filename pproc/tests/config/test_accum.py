import pytest

from pproc.config.accumulation import LegacyWindowConfig, LegacyStepAccumulation


@pytest.mark.parametrize(
    "config, grib_keys, expected",
    [
        pytest.param(
            {"windows": [{"coords": [[120], [123], [126], [129], [132], [360]]}]},
            {"mars.expver": "0001"},
            [
                {
                    "coords": [s],
                    "sequential": True,
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "metadata": {
                        "mars.expver": "0001",
                    },
                    "deaccumulate": False,
                }
                for s in [120, 123, 126, 129, 132, 360]
            ],
            id="simple",
        ),
        pytest.param(
            {"windows": [{"coords": [[0], [0, 3], [3, 6], [300, 306]]}]},
            {"timeRangeIndicator": 2},
            [
                {
                    "coords": [0],
                    "sequential": True,
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "metadata": {
                        "timeRangeIndicator": 2,
                    },
                    "deaccumulate": False,
                },
                {
                    "coords": [0, 3],
                    "sequential": True,
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "metadata": {
                        "timeRangeIndicator": 2,
                    },
                    "deaccumulate": False,
                },
                {
                    "coords": [3, 6],
                    "sequential": True,
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "metadata": {
                        "timeRangeIndicator": 2,
                    },
                    "deaccumulate": False,
                },
                {
                    "coords": [300, 306],
                    "sequential": True,
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "metadata": {
                        "timeRangeIndicator": 2,
                    },
                    "deaccumulate": False,
                },
            ],
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
            [
                {
                    "operation": "difference",
                    "coords": [a, b],
                    "sequential": True,
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "metadata": {
                        "stepType": "diff",
                        "bitsPerValue": 16,
                    },
                    "deaccumulate": False,
                }
                for a, b in [(90, 96), (93, 99), (96, 102), (270, 276)]
            ]
            + [
                {
                    "operation": "difference",
                    "coords": [a, b],
                    "sequential": True,
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_1",
                    },
                    "metadata": {
                        "timeRangeIndicator": 5,
                        "gribTablesVersionNo": 132,
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
            ],
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
            [
                {
                    "operation": "aggregation",
                    "coords": [s],
                    "sequential": True,
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "metadata": {
                        "bitsPerValue": 16,
                        "expver": "0001",
                    },
                    "deaccumulate": False,
                }
                for s in [75, 78, 81, 84, 87, 90, 288]
            ],
            id="noop-simple",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "operation": "aggregation",
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
            [
                {
                    "operation": "aggregation",
                    "coords": [a, b],
                    "sequential": True,
                    "metadata": {
                        "bitsPerValue": 16,
                        "expver": "0001",
                    },
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
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
            ],
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
            [
                {
                    "operation": "mean",
                    "coords": list(range(a, b + 1, s)),
                    "sequential": True,
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "metadata": {
                        "timeRangeIndicator": 3,
                    },
                    "deaccumulate": False,
                }
                for a, b, s in [
                    (12, 36, 6),
                    (60, 132, 6),
                    (0, 240, 6),
                    (0, 360, 24),
                ]
            ],
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
            [
                {
                    "operation": "maximum",
                    "coords": {"from": a, "to": b, "by": s},
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "sequential": True,
                    "metadata": {
                        "timeRangeIndicator": 2,
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
            ],
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
            [
                {
                    "operation": "minimum",
                    "coords": {"from": a, "to": b, "by": s},
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "sequential": True,
                    "metadata": {
                        "timeRangeIndicator": 2,
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
            ],
            id="min-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "operation": "minimum",
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
            [
                {
                    "operation": "minimum",
                    "coords": {"from": a, "to": b, "by": s},
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "sequential": True,
                    "thresholds": [{"comparison": "<=", "value": 273.15}],
                    "metadata": {
                        "type": "ep",
                        "localDefinitionNumber": 5,
                    },
                    "deaccumulate": False,
                }
                for a, b, s in [
                    (120, 240, 6),
                    (120, 168, 6),
                    (168, 240, 6),
                    (240, 360, 6),
                ]
            ],
            id="simple-threshold-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "operation": "maximum",
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
            [
                {
                    "operation": "maximum",
                    "coords": {"from": a, "to": b, "by": s},
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "sequential": True,
                    "thresholds": [
                        {"comparison": ">=", "value": 15.0},
                        {"comparison": ">=", "value": 20.0},
                        {"comparison": ">=", "value": 25.0},
                    ],
                    "metadata": {
                        "type": "ep",
                        "localDefinitionNumber": 5,
                    },
                    "deaccumulate": False,
                }
                for a, b, s in [
                    (0, 24, 6),
                    (12, 36, 6),
                    (336, 360, 6),
                ]
            ],
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
            [
                {
                    "operation": "difference",
                    "coords": [a, b],
                    "sequential": True,
                    "thresholds": [{"comparison": ">=", "value": thr} for thr in thrs],
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": f"_{i}",
                    },
                    "metadata": {
                        "type": "ep",
                        "localDefinitionNumber": 5,
                        "stepType": "accum",
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
            ]
            + [
                {
                    "operation": "difference_rate",
                    "factor": 1.0 / 24.0,
                    "coords": [a, b],
                    "sequential": True,
                    "name": {
                        "type": "default",
                        "suffix": "_2",
                        "prefix": "",
                    },
                    "thresholds": [
                        {"comparison": cmp, "value": val}
                        for cmp, vals in [("<", [0.001]), (">=", [0.003, 0.005])]
                        for val in vals
                    ],
                    "metadata": {
                        "type": "ep",
                        "localDefinitionNumber": 5,
                        "stepType": "diff",
                    },
                    "deaccumulate": False,
                }
                for a, b in [
                    (120, 240),
                    (168, 240),
                    (228, 360),
                ]
            ],
            id="diffs-threshold-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "operation": "mean",
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
            [
                {
                    "operation": "mean",
                    "coords": {"from": a, "to": b, "by": s},
                    "sequential": True,
                    "thresholds": [
                        {"comparison": "<", "value": -2},
                        {"comparison": ">=", "value": 2},
                    ],
                    "name": {
                        "type": "default",
                        "suffix": "_0",
                        "prefix": "",
                    },
                    "metadata": {
                        "type": "ep",
                        "localDefinitionNumber": 5,
                        "bitsPerValue": 24,
                    },
                    "deaccumulate": False,
                }
                for a, b, s in [
                    (120, 168, 12),
                    (168, 240, 12),
                    (240, 360, 12),
                ]
            ],
            id="mean-threshold-range",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "operation": "minimum",
                        "thresholds": [
                            {"comparison": "<", "value": -8},
                            {"comparison": "<", "value": -4},
                        ],
                        "coords": [[0], [12], [360]],
                        "metadata": {"bitsPerValue": 24},
                    },
                    {
                        "operation": "maximum",
                        "thresholds": [
                            {"comparison": ">", "value": 4},
                            {"comparison": ">", "value": 8},
                        ],
                        "coords": [[0], [12], [360]],
                        "metadata": {"bitsPerValue": 24},
                    },
                    {
                        "operation": "mean",
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
                    {
                        "operation": "maximum",
                        "std_anomaly": True,
                        "thresholds": [
                            {"comparison": ">", "value": 1},
                        ],
                        "coords": [[0], [12], [300]],
                        "metadata": {
                            "localDefinitionNumber": 30,
                            "bitsPerValue": 24,
                        },
                    },
                    {
                        "operation": "minimum",
                        "std_anomaly": True,
                        "thresholds": [
                            {"comparison": "<", "value": -1.5},
                        ],
                        "coords": [[0], [12], [300]],
                        "metadata": {
                            "localDefinitionNumber": 30,
                            "bitsPerValue": 24,
                        },
                    },
                ],
            },
            {"type": "ep", "localDefinitionNumber": 5, "bitsPerValue": 8},
            list(
                sum(
                    [
                        [
                            {
                                "operation": op,
                                "coords": [s],
                                "sequential": True,
                                "thresholds": [
                                    {"comparison": cmp, "value": val} for val in vals
                                ],
                                "name": {
                                    "type": "default",
                                    "suffix": f"_{index}",
                                    "prefix": "",
                                },
                                "metadata": {
                                    "type": "ep",
                                    "localDefinitionNumber": 5,
                                    "bitsPerValue": 24,
                                },
                                "deaccumulate": False,
                            }
                            for s in [0, 12, 360]
                        ]
                        for index, (cmp, op, vals) in enumerate(
                            [
                                ("<", "minimum", [-8, -4]),
                                (">", "maximum", [4, 8]),
                            ]
                        )
                    ],
                    [],
                )
            )
            + [
                {
                    "operation": "mean",
                    "coords": {"from": a, "to": b, "by": s},
                    "sequential": True,
                    "thresholds": [
                        {"comparison": "<", "value": -4},
                        {"comparison": ">=", "value": 2},
                    ],
                    "name": {
                        "type": "default",
                        "suffix": "_2",
                        "prefix": "",
                    },
                    "metadata": {
                        "type": "ep",
                        "localDefinitionNumber": 5,
                        "bitsPerValue": 24,
                    },
                    "deaccumulate": False,
                }
                for a, b, s in [
                    (120, 240, 12),
                    (336, 360, 12),
                ]
            ]
            + list(
                sum(
                    [
                        [
                            {
                                "operation": op,
                                "std_anomaly": True,
                                "coords": [s],
                                "sequential": True,
                                "thresholds": [{"comparison": cmp, "value": val}],
                                "name": {
                                    "type": "default",
                                    "prefix": "STDANOM_",
                                    "suffix": f"_{index + 3}",
                                },
                                "metadata": {
                                    "type": "ep",
                                    "localDefinitionNumber": 30,
                                    "bitsPerValue": 24,
                                },
                                "deaccumulate": False,
                            }
                            for s in [0, 12, 300]
                        ]
                        for index, (cmp, op, val) in enumerate(
                            [
                                (">", "maximum", 1),
                                ("<", "minimum", -1.5),
                            ]
                        )
                    ],
                    [],
                )
            ),
            id="multi-anomaly",
        ),
        pytest.param(
            {
                "windows": [
                    {
                        "operation": "mean",
                        "deaccumulate": True,
                        "coords": [[120, 123], [123, 126]],
                    }
                ]
            },
            {},
            [
                {
                    "operation": "mean",
                    "coords": [s, e],
                    "name": {
                        "type": "default",
                        "prefix": "",
                        "suffix": "_0",
                    },
                    "metadata": {},
                    "sequential": True,
                    "deaccumulate": True,
                }
                for s, e in [[120, 123], [123, 126]]
            ],
            id="deaccumulation",
        ),
    ],
)
def test_legacy_config(config, grib_keys, expected):
    configs = LegacyStepAccumulation(**config).make_configs(grib_keys)
    assert list(configs) == expected


@pytest.mark.parametrize(
    "config1, config2, merged, expected",
    [
        [
            {"operation": "difference", "coords": [[0, 6]]},
            {"operation": "minimum", "coords": [[0, 6]]},
            False,
            None,
        ],
        [
            {"operation": "difference", "coords": [[0, 6]]},
            {"operation": "difference", "coords": [[12, 18]]},
            True,
            {"operation": "difference", "coords": [[0, 6], [12, 18]]},
        ],
        [
            {"operation": "minimum", "coords": [[0, 6]], "thresholds": [{"value": 6}]},
            {
                "operation": "minimum",
                "coords": [[12, 18]],
                "thresholds": [{"value": 6}],
            },
            True,
            {
                "operation": "minimum",
                "coords": [[0, 6], [12, 18]],
                "thresholds": [{"value": 6}],
            },
        ],
        [
            {"operation": "minimum", "coords": [[0, 6]], "thresholds": [{"value": 6}]},
            {"operation": "minimum", "coords": [[0, 6]], "thresholds": [{"value": 12}]},
            True,
            {
                "operation": "minimum",
                "coords": [[0, 6]],
                "thresholds": [{"value": 6}, {"value": 12}],
            },
        ],
        [
            {"operation": "minimum", "coords": [[0, 6]], "thresholds": [{"value": 6}]},
            {
                "operation": "minimum",
                "coords": [[12, 18]],
                "thresholds": [{"value": 12}],
            },
            False,
            None,
        ],
    ],
    ids=[
        "diff-op",
        "merge-steps",
        "threshold-merge-steps",
        "threshold-merge",
        "no-merge",
    ],
)
def test_legacy_merge(config1, config2, merged, expected):
    accum1 = LegacyWindowConfig(**config1)
    accum2 = LegacyWindowConfig(**config2)
    assert accum1.merge(accum2) == merged
    if merged:
        assert accum1 == LegacyWindowConfig(**expected)


@pytest.mark.parametrize(
    "config, expected",
    [
        [
            {
                "windows": [
                    {
                        "operation": "minimum",
                        "coords": [[0]],
                        "thresholds": [{"metadata": {"paramId": 1}}],
                    }
                ]
            },
            [{"param": [1], "step": [0]}],
        ],
        [
            {
                "windows": [
                    {
                        "operation": "minimum",
                        "coords": [[0, 6]],
                        "thresholds": [{"metadata": {"paramId": 1}}],
                    }
                ]
            },
            [{"param": [1], "step": ["0-6"]}],
        ],
        [
            {
                "windows": [
                    {
                        "operation": "minimum",
                        "coords": [[0]],
                        "thresholds": [{"metadata": {"paramId": 1}}],
                    },
                    {
                        "operation": "minimum",
                        "coords": [[6]],
                        "thresholds": [{"metadata": {"paramId": 1}}],
                    },
                ]
            },
            [{"param": [1], "step": [0]}, {"param": [1], "step": [6]}],
        ],
        [
            {
                "windows": [
                    {
                        "operation": "minimum",
                        "coords": [[0], [6]],
                        "thresholds": [{"metadata": {"paramId": 1}}],
                    },
                ]
            },
            [{"param": [1], "step": [0, 6]}],
        ],
        [
            {
                "windows": [
                    {
                        "operation": "minimum",
                        "coords": [[0]],
                        "thresholds": [
                            {"metadata": {"paramId": 1}},
                            {"metadata": {"paramId": 2}},
                        ],
                    }
                ]
            },
            [{"param": [1, 2], "step": [0]}],
        ],
        [
            {
                "windows": [
                    {
                        "operation": "minimum",
                        "coords": [[0]],
                        "thresholds": [{"metadata": {"paramId": 1}}],
                    },
                    {
                        "operation": "minimum",
                        "std_anomaly": True,
                        "coords": [[0]],
                        "thresholds": [{"metadata": {"paramId": 2}}],
                    },
                    {
                        "operation": "minimum",
                        "coords": [[6]],
                        "thresholds": [{"metadata": {"paramId": 2}}],
                    },
                ],
            },
            [
                {"param": [1], "step": [0]},
                {"param": [2], "step": [0]},
                {"param": [2], "step": [6]},
            ],
        ],
    ],
    ids=["simple", "range", "windows", "multi-step", "multi-threshold", "anomaly"],
)
def test_legacy_out_mars(config, expected):
    config = LegacyStepAccumulation(**config)
    assert config.out_mars(dim="step") == expected
