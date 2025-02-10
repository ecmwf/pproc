import os
import pytest

from pproc.config.schema import Schema
from pproc.config import types
from pproc.config.utils import deep_update, extract_mars, expand

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def transform_request(request: dict) -> dict:
    if isinstance(request["param"], int):
        request["param"] = str(request["param"])
    return request


def check_request_equality(reqs1, reqs2):
    for i, req in enumerate(reqs1):
        assert transform_request(req) == transform_request(reqs2[i])


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
                        "request": [
                            {
                                "class": "od",
                                "stream": "enfo",
                                "expver": "0001",
                                "levtype": "pl",
                                "domain": "g",
                                "param": param,
                                "date": "20241001",
                                "time": "0",
                                "type": "cf",
                            },
                            {
                                "class": "od",
                                "stream": "enfo",
                                "expver": "0001",
                                "levtype": "pl",
                                "domain": "g",
                                "param": param,
                                "date": "20241001",
                                "time": "0",
                                "type": "pf",
                            },
                        ],
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
                                        {"range": [0, 168, 12]},
                                        {"range": [24, 192, 12]},
                                    ],
                                },
                                {
                                    "window_operation": "standard_deviation",
                                    "grib_set": {"type": "fcstdev"},
                                    "periods": [
                                        {"range": [0, 168, 12]},
                                        {"range": [24, 192, 12]},
                                    ],
                                },
                                {
                                    "window_operation": "minimum",
                                    "grib_set": {"type": "fcmin"},
                                    "periods": [
                                        {"range": [0, 168, 12]},
                                        {"range": [24, 192, 12]},
                                    ],
                                },
                                {
                                    "window_operation": "maximum",
                                    "grib_set": {"type": "fcmax"},
                                    "periods": [
                                        {"range": [0, 168, 12]},
                                        {"range": [24, 192, 12]},
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
        {
            "outputs": {
                "std": {"target": {"type": "null"}},
            },
        },
        types.EnsmsConfig,
        {
            "outputs": {
                "mean": {"metadata": {"type": "em"}, "target": {"type": "fdb"}},
                "std": {"metadata": {"type": "es"}, "target": {"type": "null"}},
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
                "228": {
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
                                **DEFAULT_REQUEST,
                                "levtype": "sfc",
                                "param": "228",
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
                                        {"range": [0, 168, 12]},
                                        {"range": [24, 192, 12]},
                                    ],
                                },
                                {
                                    "window_operation": "standard_deviation",
                                    "grib_set": {"type": "fcstdev"},
                                    "periods": [
                                        {"range": [0, 168, 12]},
                                        {"range": [24, 192, 12]},
                                    ],
                                },
                                {
                                    "window_operation": "minimum",
                                    "grib_set": {"type": "fcmin"},
                                    "periods": [
                                        {"range": [0, 168, 12]},
                                        {"range": [24, 192, 12]},
                                    ],
                                },
                                {
                                    "window_operation": "maximum",
                                    "grib_set": {"type": "fcmax"},
                                    "periods": [
                                        {"range": [0, 168, 12]},
                                        {"range": [24, 192, 12]},
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
                "param": "172228",
                "stream": "msmm",
                "type": ["fcmean", "fcstdev", "fcmax"],
                "fcmonth": [1, 2, 3],
            },
            "228",
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

    overrides, cfg_type, updates = TEST_CASES[request.node.callspec.id]
    config = types.ConfigFactory.from_outputs(schema, [output_request], **overrides)
    assert type(config) == cfg_type
    check = default_config(input_param)
    deep_update(check, updates)
    assert config.model_dump(by_alias=True) == cfg_type(**check).model_dump(
        by_alias=True
    )
    output_types = (
        [output_request["type"]]
        if isinstance(output_request["type"], str)
        else output_request["type"]
    )
    check_request_equality(
        config.out_mars(),
        [
            {
                **output_request,
                **config._set_number({"type": tp}),
                "target": "fdb",
                **extract_mars(config.outputs.default.metadata),
                "type": tp,
            }
            for tp in output_types
        ],
    )


@pytest.mark.parametrize(
    "entrypoint, input_request, periods",
    [
        [
            "pproc-accumulate",
            [
                {
                    **DEFAULT_REQUEST,
                    "levelist": [250, 500],
                    "step": list(range(0, 193, 6)),
                    "type": "cf",
                },
                {
                    **DEFAULT_REQUEST,
                    "levelist": [250, 500],
                    "step": list(range(0, 193, 6)),
                    "type": "pf",
                },
            ],
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
            [
                {
                    **DEFAULT_REQUEST,
                    "levelist": [250, 500],
                    "step": list(range(0, 193, 6)),
                    "type": "cf",
                },
                {
                    **DEFAULT_REQUEST,
                    "levelist": [250, 500],
                    "step": list(range(0, 193, 6)),
                    "type": "pf",
                },
            ],
            {
                "type": "ranges",
                "from": "0",
                "to": "192",
                "interval": "6",
            },
        ],
        [
            "pproc-monthly-stats",
            [
                {
                    **DEFAULT_REQUEST,
                    "levtype": "sfc",
                    "param": "228",
                    "stream": "mmsf",
                    "type": "fc",
                    "step": list(range(0, 2209, 24)),
                }
            ],
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
            [
                {
                    **DEFAULT_REQUEST,
                    "levelist": [250, 500],
                    "step": list(range(0, 193, 6)),
                    "type": "cf",
                },
                {
                    **DEFAULT_REQUEST,
                    "levelist": [250, 500],
                    "step": list(range(0, 193, 6)),
                    "type": "pf",
                },
            ],
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

    overrides, cfg_type, updates = TEST_CASES[request.node.callspec.id]
    config = types.ConfigFactory.from_inputs(
        schema, entrypoint, input_request, **overrides
    )
    assert type(config) == cfg_type
    check = default_config(input_request[0]["param"])
    windows = updates["parameters"][input_request[0]["param"]]["accumulations"]["step"][
        "windows"
    ]
    for window in windows:
        window["periods"] = periods
    deep_update(check, updates)
    assert config.model_dump(exclude_defaults=True, by_alias=True) == cfg_type(
        **check
    ).model_dump(exclude_defaults=True, by_alias=True)
    check_request_equality(
        config.in_mars(),
        [{"source": "fdb", **config._set_number(req)} for req in input_request],
    )


def test_wind():
    schema = Schema(os.path.join(TEST_DIR, "schema.yaml"))

    input_request = [
        {
            **DEFAULT_REQUEST,
            "levelist": [250, 500],
            "step": list(range(0, 193, 6)),
            "type": ["cf", "pf"],
            "param": [165, 166],
        },
    ]
    config = types.ConfigFactory.from_inputs(schema, "pproc-ensms", input_request)
    check_request_equality(
        config.out_mars(),
        sum(
            [
                [
                    {
                        "target": "fdb",
                        **DEFAULT_REQUEST,
                        "levelist": [250, 500],
                        "step": list(map(str, range(0, 193, 6))),
                        "type": tp,
                        "param": param,
                    }
                    for tp in ["em", "es"]
                ]
                for param in [165, 166, 207]
            ],
            [],
        ),
    )
