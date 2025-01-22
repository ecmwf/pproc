import os
import pytest

from pproc.config.schema import Schema
from pproc.config import types
from pproc.config.utils import deep_update

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_REQUEST = {
    "class": "od",
    "stream": "enfo",
    "expver": "0001",
    "levtype": "pl",
    "domain": "g",
    "param": "130",
    "date": "20241001",
    "time": "0",
}


def default_config(param: str):
    return {
        "members": 50,
        "total_fields": 51,
        "sources": {"fc": {"type": "fdb"}},
        "outputs": {
            "default": {"target": {"type": "fdb"}},
        },
        "parameters": {
            param: {
                "sources": {
                    "fc": {
                        "request": {
                            "class": "od",
                            "stream": "enfo",
                            "expver": "0001",
                            "levtype": "pl",
                            "domain": "g",
                            "param": param,
                            "date": "20241001",
                            "time": "0",
                            "type": ["cf", "pf"],
                        }
                    }
                },
                "accumulations": {
                    "step": {"type": "legacywindow"},
                },
            }
        },
    }


TEST_CASES = {
    "accumulate": [
        {},
        types.AccumConfig,
        {
            "outputs": {
                "accum": {"metadata": {}, "target": {"type": "fdb"}},
            },
            "parameters": {
                "130": {
                    "accumulations": {
                        "levelist": {"coords": [[250], [500]]},
                        "step": {
                            "type": "legacywindow",
                            "windows": [
                                {
                                    "window_operation": "mean",
                                    "grib_set": {"type": "fcmean"},
                                    "periods": [
                                        {"range": [0, 168, 6]},
                                        {"range": [24, 192, 6]},
                                    ],
                                },
                                {
                                    "window_operation": "standard_deviation",
                                    "grib_set": {"type": "fcstdev"},
                                    "periods": [
                                        {"range": [0, 168, 6]},
                                        {"range": [24, 192, 6]},
                                    ],
                                },
                                {
                                    "window_operation": "minimum",
                                    "grib_set": {"type": "fcmin"},
                                    "periods": [
                                        {"range": [0, 168, 6]},
                                        {"range": [24, 192, 6]},
                                    ],
                                },
                                {
                                    "window_operation": "maximum",
                                    "grib_set": {"type": "fcmax"},
                                    "periods": [
                                        {"range": [0, 168, 6]},
                                        {"range": [24, 192, 6]},
                                    ],
                                },
                            ],
                        },
                    }
                }
            },
        },
    ],
    "ensms": [
        {},
        types.EnsmsConfig,
        {
            "outputs": {
                "mean": {"metadata": {"type": "em"}, "target": {"type": "fdb"}},
                "std": {"metadata": {"type": "es"}, "target": {"type": "fdb"}},
            },
            "members": 50,
            "total_fields": 51,
            "parameters": {
                "130": {
                    "metadata": {"bitsPerValue": 16, "perturbationNumber": 0},
                    "accumulations": {
                        "levelist": {"coords": [[250], [500]]},
                        "step": {
                            "type": "legacywindow",
                            "windows": [
                                {
                                    "window_operation": "none",
                                    "grib_set": {},
                                    "periods": [
                                        {"range": [0, 168, 6]},
                                        {"range": [24, 192, 6]},
                                    ],
                                }
                            ],
                        },
                    },
                }
            },
        },
    ],
    "monthly-stats": [
        {},
        types.MonthlyStatsConfig,
        {
            "outputs": {
                "stats": {"target": {"type": "fdb"}},
            },
            "members": {"start": 0, "end": 50},
            "total_fields": 51,
            "parameters": {
                "228.128": {
                    "preprocessing": [
                        {
                            "operation": "scale",
                            "value": 0.00001157407,
                        }
                    ],
                    "vmin": 0.0,
                    "sources": {
                        "fc": {
                            "request": {
                                "levtype": "sfc",
                                "param": "228.128",
                                "stream": "mmsf",
                                "type": "fc",
                            }
                        }
                    },
                    "metadata": {
                        "paramId": 172228,
                        "stream": "msmm",
                    },
                    "accumulations": {
                        "step": {
                            "type": "legacywindow",
                            "windows": [
                                {
                                    "window_operation": "mean",
                                    "grib_set": {"type": "fcmean"},
                                    "include_start_step": True,
                                    "deaccumulate": True,
                                    "periods": [
                                        {"range": [0, 744, 24]},
                                        {"range": [744, 1464, 24]},
                                        {"range": [1464, 2208, 24]},
                                    ],
                                },
                                {
                                    "window_operation": "standard_deviation",
                                    "grib_set": {"type": "fcstdev"},
                                    "include_start_step": True,
                                    "deaccumulate": True,
                                    "periods": [
                                        {"range": [0, 744, 24]},
                                        {"range": [744, 1464, 24]},
                                        {"range": [1464, 2208, 24]},
                                    ],
                                },
                                {
                                    "window_operation": "maximum",
                                    "grib_set": {"type": "fcmax"},
                                    "include_start_step": True,
                                    "deaccumulate": True,
                                    "periods": [
                                        {"range": [0, 744, 24]},
                                        {"range": [744, 1464, 24]},
                                        {"range": [1464, 2208, 24]},
                                    ],
                                },
                            ],
                        }
                    },
                }
            },
        },
    ],
    "with-overrides": [
        {
            "parallelisation": {"n_par_compute": 2},
            "outputs": {"default": {"metadata": {"expver": "9999"}}},
        },
        types.AccumConfig,
        {
            "outputs": {
                "default": {
                    "metadata": {"expver": "9999"},
                    "target": {"type": "fdb"},
                },
                "accum": {"metadata": {}, "target": {"type": "fdb"}},
            },
            "parallelisation": {"n_par_compute": 2},
            "parameters": {
                "130": {
                    "accumulations": {
                        "levelist": {"coords": [[250], [500]]},
                        "step": {
                            "type": "legacywindow",
                            "windows": [
                                {
                                    "window_operation": "mean",
                                    "grib_set": {"type": "fcmean"},
                                    "periods": [
                                        {"range": [0, 168, 6]},
                                        {"range": [24, 192, 6]},
                                    ],
                                },
                                {
                                    "window_operation": "standard_deviation",
                                    "grib_set": {"type": "fcstdev"},
                                    "periods": [
                                        {"range": [0, 168, 6]},
                                        {"range": [24, 192, 6]},
                                    ],
                                },
                                {
                                    "window_operation": "minimum",
                                    "grib_set": {"type": "fcmin"},
                                    "periods": [
                                        {"range": [0, 168, 6]},
                                        {"range": [24, 192, 6]},
                                    ],
                                },
                                {
                                    "window_operation": "maximum",
                                    "grib_set": {"type": "fcmax"},
                                    "periods": [
                                        {"range": [0, 168, 6]},
                                        {"range": [24, 192, 6]},
                                    ],
                                },
                            ],
                        },
                    }
                }
            },
        },
    ],
}


@pytest.mark.parametrize(
    "output_request, input_param",
    [
        [
            {
                **DEFAULT_REQUEST,
                "levelist": [250, 500],
                "step": ["0-168", "24-192"],
                "type": ["fcmean", "fcstdev", "fcmin", "fcmax"],
            },
            "130",
        ],
        [
            {
                **DEFAULT_REQUEST,
                "levelist": [250, 500],
                "step": ["0-168", "24-192"],
                "type": "em",
            },
            "130",
        ],
        [
            {
                **DEFAULT_REQUEST,
                "levtype": "sfc",
                "param": "228.172",
                "stream": "msmm",
                "type": ["fcmean", "fcstdev", "fcmax"],
                "forecastMonth": [1, 2, 3],
            },
            "228.128",
        ],
        [
            {
                **DEFAULT_REQUEST,
                "levelist": [250, 500],
                "step": ["0-168", "24-192"],
                "type": ["fcmean", "fcstdev", "fcmin", "fcmax"],
            },
            "130",
        ],
    ],
    ids=TEST_CASES.keys(),
)
def test_factory_from_outputs(request, output_request, input_param):
    schema = Schema(os.path.join(TEST_DIR, "schema.yaml"))

    overrides, cfg_type, expected = TEST_CASES[request.node.callspec.id]
    output_types = output_request.get("type")
    requests = (
        [output_request]
        if not isinstance(output_types, list)
        else [{**output_request, "type": t} for t in output_types]
    )
    config = types.ConfigFactory.from_outputs(schema, requests, **overrides)
    assert type(config) == cfg_type
    check = default_config(input_param)
    deep_update(check, expected)
    assert config.model_dump(exclude_defaults=True, by_alias=True) == cfg_type(
        **check
    ).model_dump(exclude_defaults=True, by_alias=True)


@pytest.mark.parametrize(
    "entrypoint, input_request, periods",
    [
        [
            "pproc-accumulate",
            {
                **DEFAULT_REQUEST,
                "levelist": [250, 500],
                "step": list(range(0, 193, 6)),
                "type": ["cf", "pf"],
            },
            {
                "type": "ranges",
                "from": "0",
                "to": "192",
                "by": "6",
                "interval": 24,
                "width": 168,
            },
        ],
        [
            "pproc-ensms",
            {
                **DEFAULT_REQUEST,
                "levelist": [250, 500],
                "step": list(range(0, 193, 6)),
                "type": ["cf", "pf"],
            },
            {
                "type": "ranges",
                "from": "0",
                "to": "192",
                "interval": "6",
            },
        ],
        [
            "pproc-monthly-stats",
            {
                **DEFAULT_REQUEST,
                "levtype": "sfc",
                "param": "228.128",
                "stream": "mmsf",
                "type": "fc",
                "step": list(range(0, 2209, 24)),
            },
            {
                "type": "monthly",
                "date": "20241001",
                "from": "0",
                "to": "2208",
                "by": "24",
            },
        ],
        [
            "pproc-accumulate",
            {
                **DEFAULT_REQUEST,
                "levelist": [250, 500],
                "step": list(range(0, 193, 6)),
                "type": ["cf", "pf"],
            },
            {
                "type": "ranges",
                "from": "0",
                "to": "192",
                "by": "6",
                "interval": 24,
                "width": 168,
            },
        ],
    ],
    ids=TEST_CASES.keys(),
)
def test_factory_from_inputs(request, entrypoint, input_request, periods):
    schema = Schema(os.path.join(TEST_DIR, "schema.yaml"))

    overrides, cfg_type, expected = TEST_CASES[request.node.callspec.id]
    windows = expected["parameters"][input_request["param"]]["accumulations"]["step"][
        "windows"
    ]
    for window in windows:
        window["periods"] = periods
    config = types.ConfigFactory.from_inputs(
        schema, entrypoint, [input_request], **overrides
    )
    assert type(config) == cfg_type
    check = default_config(input_request["param"])
    deep_update(check, expected)
    assert config.model_dump(exclude_defaults=True, by_alias=True) == cfg_type(
        **check
    ).model_dump(exclude_defaults=True, by_alias=True)


@pytest.mark.parametrize(
    "config, expected",
    [
        [
            {
                "members": 5,
                "sources": {"default": {"type": "fdb"}},
                "parameters": {
                    "2t": {
                        "sources": {
                            "fc": {
                                "request": {
                                    "param": "228.128",
                                }
                            },
                            "clim": {"request": {"param": "228004"}},
                        },
                        "accumulations": {
                            "step": {
                                "type": "default",
                                "coords": [["0-24"], ["48-72"]],
                            },
                        },
                        "clim": {
                            "accumulations": {
                                "date": {
                                    "operation": "mean",
                                    "coords": [["20241001", "20241002"]],
                                },
                            },
                        },
                    }
                },
            },
            [
                {
                    "param": "228.128",
                    "step": ["0-24", "48-72"],
                    "source": "fdb",
                },
                {
                    "param": "228004",
                    "step": ["0-24", "48-72"],
                    "date": ["20241001", "20241002"],
                    "source": "fdb",
                },
            ],
        ],
    ],
    ids=["multi-param"],
)
def test_inputs(config: dict, expected: list[dict]):
    config = types.AnomalyConfig(**config)
    assert config.inputs() == expected
