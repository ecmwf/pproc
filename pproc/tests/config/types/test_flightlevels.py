import copy
import pytest
from contextlib import nullcontext

from pproc.config.types import FlightLevelsConfig
from pproc.config.utils import deep_update

BASE_OUTPUT = {
    "param": [260290],
    "levtype": "fl",
    "levelist": [110, 200, 300, 340, 380],
    "step": [0, 3, 6],
    "target": "fdb",
}


@pytest.mark.parametrize(
    "configs, merged_or_error",
    [
        [[{}, {"target_flight_levels": [110, 200]}], ValueError],
        [[{}, {"target_flight_levels": [110, 159]}], ValueError],
        [
            [
                {},
                {
                    "parameters": {
                        "cat": {
                            "inputs": {
                                "fc": {
                                    "request": {
                                        "stream": "enfo",
                                        "type": "pf",
                                        "number": [1, 2, 3],
                                    }
                                },
                                "lnsp": {
                                    "request": {
                                        "stream": "enfo",
                                        "type": "pf",
                                        "number": [1, 2, 3],
                                    }
                                },
                            }
                        }
                    }
                },
            ],
            {
                "parameters": {
                    "cat": {
                        "inputs": {
                            "fc": {
                                "request": [
                                    {
                                        "stream": "oper",
                                        "type": "fc",
                                        "param": 260290,
                                        "levtype": "ml",
                                        "levelist": list(range(1, 138)),
                                    },
                                    {
                                        "stream": "enfo",
                                        "type": "pf",
                                        "param": 260290,
                                        "levtype": "ml",
                                        "levelist": list(range(1, 138)),
                                        "number": [1, 2, 3],
                                    },
                                ]
                            },
                            "lnsp": {
                                "request": [
                                    {
                                        "stream": "oper",
                                        "type": "fc",
                                        "param": 152,
                                        "levtype": "ml",
                                        "levelist": 1,
                                    },
                                    {
                                        "stream": "enfo",
                                        "type": "pf",
                                        "param": 152,
                                        "levtype": "ml",
                                        "levelist": 1,
                                        "number": [1, 2, 3],
                                    },
                                ],
                            },
                        }
                    }
                }
            },
        ],
        [
            [
                {},
                {
                    "parameters": {
                        "cat": {
                            "inputs": {
                                "fc": {"request": {"param": 130}},
                            }
                        }
                    }
                },
            ],
            {
                "parameters": {
                    "cat": {
                        "inputs": {
                            "fc": {
                                "request": [
                                    {
                                        "stream": "oper",
                                        "type": "fc",
                                        "param": 260290,
                                        "levtype": "ml",
                                        "levelist": list(range(1, 138)),
                                    },
                                    {
                                        "stream": "oper",
                                        "type": "fc",
                                        "param": 130,
                                        "levtype": "ml",
                                        "levelist": list(range(1, 138)),
                                    },
                                ],
                            },
                            "lnsp": {
                                "request": [
                                    {
                                        "stream": "oper",
                                        "type": "fc",
                                        "param": 152,
                                        "levtype": "ml",
                                        "levelist": 1,
                                    },
                                ],
                            },
                        },
                    }
                }
            },
        ],
        [
            [
                {},
                {
                    "parameters": {
                        "cat": {
                            "inputs": {
                                "fc": {"request": {"stream": "enfo", "type": "cf"}},
                            }
                        }
                    }
                },
            ],
            AssertionError,
        ],
    ],
    ids=["diff-levels", "invalid-level", "fc-ens", "multi-param", "diff-fcs"],
)
def test_merge(configs, merged_or_error):
    base_config = {
        "target_flight_levels": [110, 200, 300, 340, 380],
        "parameters": {
            "cat": {
                "inputs": {
                    "fc": {
                        "request": {
                            "stream": "oper",
                            "type": "fc",
                            "param": 260290,
                            "levtype": "ml",
                            "levelist": list(range(1, 138)),
                        },
                    },
                    "lnsp": {
                        "request": {
                            "stream": "oper",
                            "type": "fc",
                            "param": 152,
                            "levtype": "ml",
                            "levelist": 1,
                        },
                    },
                },
            },
        },
        "inputs": {"default": {"source": {"type": "fdb"}}},
        "outputs": {
            "default": {"target": {"type": "fdb"}},
        },
    }
    config = FlightLevelsConfig(**deep_update(copy.deepcopy(base_config), configs[0]))

    if isinstance(merged_or_error, dict):
        context = nullcontext()
    else:
        context = pytest.raises(merged_or_error)

    with context:
        for other in configs[1:]:
            other_config = FlightLevelsConfig(
                **deep_update(copy.deepcopy(base_config), other)
            )
            config = config.merge(other_config)
        expected = FlightLevelsConfig(
            **deep_update(copy.deepcopy(base_config), merged_or_error)
        )
        assert config == expected


@pytest.mark.parametrize(
    "inputs, expected",
    [
        [
            [
                {
                    "stream": "oper",
                    "type": "fc",
                    "param": [260290],
                    "levtype": "ml",
                    "levelist": list(range(1, 138)),
                },
            ],
            [
                {
                    **BASE_OUTPUT,
                    "stream": "oper",
                    "type": "fc",
                },
            ],
        ],
        [
            [
                {
                    "stream": "oper",
                    "type": "fc",
                    "param": [260290],
                    "levtype": "ml",
                    "levelist": list(range(1, 138)),
                },
                {
                    "stream": "enfo",
                    "type": "pf",
                    "param": [260290],
                    "levtype": "ml",
                    "levelist": list(range(1, 138)),
                    "number": [1, 2, 3],
                },
            ],
            [
                {
                    **BASE_OUTPUT,
                    "stream": "oper",
                    "type": "fc",
                },
                {
                    **BASE_OUTPUT,
                    "stream": "enfo",
                    "type": "pf",
                    "number": [1, 2, 3],
                },
            ],
        ],
        [
            [
                {
                    "stream": "oper",
                    "type": "fc",
                    "param": [130, 260290],
                    "levtype": "ml",
                    "levelist": list(range(1, 138)),
                },
                {
                    "stream": "enfo",
                    "type": "pf",
                    "param": [130, 260290],
                    "levtype": "ml",
                    "levelist": list(range(1, 138)),
                    "number": [1, 2, 3],
                },
            ],
            [
                {
                    **BASE_OUTPUT,
                    "stream": "oper",
                    "type": "fc",
                    "param": [130, 260290],
                },
                {
                    **BASE_OUTPUT,
                    "stream": "enfo",
                    "type": "pf",
                    "param": [130, 260290],
                    "number": [1, 2, 3],
                },
            ],
        ],
        [
            [
                {
                    "stream": "oper",
                    "type": "fc",
                    "param": [260290],
                    "levtype": "ml",
                    "levelist": list(range(1, 138)),
                    "timespan": "none",
                },
            ],
            [
                {
                    **BASE_OUTPUT,
                    "stream": "oper",
                    "type": "fc",
                    "timespan": "none",
                },
            ],
        ],
    ],
    ids=["fc", "ens", "multi-param", "fc-timespan"],
)
def test_outputs(inputs, expected):
    config = FlightLevelsConfig(
        target_flight_levels=[110, 200, 300, 340, 380],
        parameters={
            "cat": {
                "inputs": {
                    "fc": {"request": inputs},
                    "lnsp": {
                        "request": [{**x, "param": 152, "levelist": 1} for x in inputs]
                    },
                },
                "num_inputs": 1,
                "accumulations": {
                    "step": {
                        "coords": [[0], [3], [6]],
                    }
                },
            }
        },
        inputs={"default": {"source": {"type": "fdb"}}},
        outputs={
            "default": {"target": {"type": "fdb"}},
        },
    )
    config.print()
    assert sum(
        [
            [{**x, "param": param, "step": [0, 3, 6], "source": "fdb"} for x in inputs]
            for param in inputs[0]["param"]
        ],
        [],
    ) + [
        {**x, "param": 152, "step": [0, 3, 6], "levelist": 1, "source": "fdb"}
        for x in inputs
    ] == list(
        config.in_mars(sources=["fdb"])
    )
    outputs = list(config.out_mars(targets=["fdb"]))
    assert outputs == expected


def test_from_schema():
    config = FlightLevelsConfig.from_schema(
        {
            "inputs": [
                {
                    "stream": "oper",
                    "type": "fc",
                    "param": 260290,
                    "levtype": "ml",
                    "levelist": list(range(1, 138)),
                },
                {
                    "stream": "oper",
                    "type": "fc",
                    "param": 152,
                    "levtype": "ml",
                    "levelist": 1,
                },
            ],
            "target_flight_levels": [110, 200, 300, 340, 380],
        }
    )
    assert config == FlightLevelsConfig(
        target_flight_levels=[110, 200, 300, 340, 380],
        parameters={
            "260290": {
                "inputs": {
                    "fc": {
                        "request": {
                            "stream": "oper",
                            "type": "fc",
                            "param": 260290,
                            "levtype": "ml",
                            "levelist": list(range(1, 138)),
                        },
                    },
                    "lnsp": {
                        "request": {
                            "stream": "oper",
                            "type": "fc",
                            "param": 152,
                            "levtype": "ml",
                            "levelist": 1,
                        },
                    },
                },
            },
        },
        inputs={"default": {"source": {"type": "fdb"}}},
        outputs={
            "default": {"target": {"type": "fdb"}},
        },
    )
