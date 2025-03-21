import os
import pytest

from pproc.schema.schema import Schema
from pproc.config import types
from pproc.config.utils import deep_update, extract_mars, expand
from pproc.config.factory import ConfigFactory

from conftest import schema


def transform_request(request: dict) -> dict:
    if isinstance(request["param"], int):
        request["param"] = str(request["param"])
    return request


def check_request_equality(reqs1, reqs2):
    assert len(reqs1) == len(reqs2)
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
                                "number": list(range(1, 51)),
                            },
                        ],
                    }
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
                    "dtype": "float64",
                    "accumulations": {
                        "levelist": {"coords": [[250], [500]]},
                        "step": {
                            "type": "legacywindow",
                            "windows": [
                                {
                                    "operation": "mean",
                                    "grib_keys": {"type": "fcmean"},
                                    "coords": [
                                        {"from": 0, "to": 168, "by": 12},
                                        {"from": 24, "to": 192, "by": 12},
                                    ],
                                },
                                {
                                    "operation": "standard_deviation",
                                    "grib_keys": {"type": "fcstdev"},
                                    "coords": [
                                        {"from": 0, "to": 168, "by": 12},
                                        {"from": 24, "to": 192, "by": 12},
                                    ],
                                },
                                {
                                    "operation": "minimum",
                                    "grib_keys": {"type": "fcmin"},
                                    "coords": [
                                        {"from": 0, "to": 168, "by": 12},
                                        {"from": 24, "to": 192, "by": 12},
                                    ],
                                },
                                {
                                    "operation": "maximum",
                                    "grib_keys": {"type": "fcmax"},
                                    "coords": [
                                        {"from": 0, "to": 168, "by": 12},
                                        {"from": 24, "to": 192, "by": 12},
                                    ],
                                },
                            ],
                        },
                    },
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
            "total_fields": 51,
            "parameters": {
                "130": {
                    "dtype": "float64",
                    "metadata": {"bitsPerValue": 16, "perturbationNumber": 0},
                    "accumulations": {
                        "levelist": {"coords": [[250], [500]]},
                        "step": {"coords": [[x] for x in range(0, 169, 6)]},
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
            "total_fields": 51,
            "parameters": {
                "228": {
                    "dtype": "float64",
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
                                "number": list(range(0, 51)),
                            }
                        }
                    },
                    "metadata": {
                        "paramId": 172228,
                        "stream": "msmm",
                        "bitsPerValue": 16,
                    },
                    "accumulations": {
                        "step": {
                            "type": "legacywindow",
                            "windows": [
                                {
                                    "operation": "mean",
                                    "grib_keys": {"type": "fcmean"},
                                    "include_start": True,
                                    "deaccumulate": True,
                                    "coords": [
                                        {"from": 0, "to": 744, "by": 24},
                                        {"from": 744, "to": 1464, "by": 24},
                                        {"from": 1464, "to": 2208, "by": 24},
                                    ],
                                },
                                {
                                    "operation": "standard_deviation",
                                    "grib_keys": {"type": "fcstdev"},
                                    "include_start": True,
                                    "deaccumulate": True,
                                    "coords": [
                                        {"from": 0, "to": 744, "by": 24},
                                        {"from": 744, "to": 1464, "by": 24},
                                        {"from": 1464, "to": 2208, "by": 24},
                                    ],
                                },
                                {
                                    "operation": "maximum",
                                    "grib_keys": {"type": "fcmax"},
                                    "include_start": True,
                                    "deaccumulate": True,
                                    "coords": [
                                        {"from": 0, "to": 744, "by": 24},
                                        {"from": 744, "to": 1464, "by": 24},
                                        {"from": 1464, "to": 2208, "by": 24},
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
                    "dtype": "float64",
                    "accumulations": {
                        "levelist": {"coords": [[250], [500]]},
                        "step": {
                            "type": "legacywindow",
                            "windows": [
                                {
                                    "operation": "mean",
                                    "grib_keys": {"type": "fcmean"},
                                    "coords": [
                                        {"from": 0, "to": 168, "by": 12},
                                        {"from": 24, "to": 192, "by": 12},
                                    ],
                                },
                                {
                                    "operation": "standard_deviation",
                                    "grib_keys": {"type": "fcstdev"},
                                    "coords": [
                                        {"from": 0, "to": 168, "by": 12},
                                        {"from": 24, "to": 192, "by": 12},
                                    ],
                                },
                                {
                                    "operation": "minimum",
                                    "grib_keys": {"type": "fcmin"},
                                    "coords": [
                                        {"from": 0, "to": 168, "by": 12},
                                        {"from": 24, "to": 192, "by": 12},
                                    ],
                                },
                                {
                                    "operation": "maximum",
                                    "grib_keys": {"type": "fcmax"},
                                    "coords": [
                                        {"from": 0, "to": 168, "by": 12},
                                        {"from": 24, "to": 192, "by": 12},
                                    ],
                                },
                            ],
                        },
                    },
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
                "number": list(range(0, 51)),
            },
            "130",
        ],
        [
            {
                **DEFAULT_REQUEST,
                "levelist": [250, 500],
                "step": list(range(0, 169, 6)),
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
                "number": list(range(0, 51)),
            },
            "228",
        ],
        [
            {
                **DEFAULT_REQUEST,
                "levelist": [250, 500],
                "step": ["0-168", "24-192"],
                "type": ["fcmean", "fcstdev", "fcmin", "fcmax"],
                "number": list(range(0, 51)),
            },
            "130",
        ],
    ],
    ids=TEST_CASES.keys(),
)
def test_from_outputs(request, output_request, input_param):
    test_schema = Schema(schema())

    overrides, cfg_type, updates = TEST_CASES[request.node.callspec.id]
    config = ConfigFactory.from_outputs(test_schema, [output_request], **overrides)
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
        list(config.out_mars()),
        [
            {
                **output_request,
                "target": "fdb",
                **extract_mars(config.outputs.default.metadata),
                "type": tp,
            }
            for tp in output_types
        ],
    )


@pytest.mark.parametrize(
    "entrypoint, input_request, step_accum, stepby",
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
                    "number": list(range(1, 51)),
                },
            ],
            {
                "type": "legacywindow",
                "windows": [
                    {
                        "operation": "mean",
                        "grib_keys": {"type": "fcmean"},
                        "coords": [
                            {"from": 0, "to": 168, "by": 12},
                            {"from": 24, "to": 192, "by": 12},
                        ],
                    },
                    {
                        "operation": "standard_deviation",
                        "grib_keys": {"type": "fcstdev"},
                        "coords": [
                            {"from": 0, "to": 168, "by": 12},
                            {"from": 24, "to": 192, "by": 12},
                        ],
                    },
                    {
                        "operation": "minimum",
                        "grib_keys": {"type": "fcmin"},
                        "coords": [
                            {"from": 0, "to": 168, "by": 12},
                            {"from": 24, "to": 192, "by": 12},
                        ],
                    },
                    {
                        "operation": "maximum",
                        "grib_keys": {"type": "fcmax"},
                        "coords": [
                            {"from": 0, "to": 168, "by": 12},
                            {"from": 24, "to": 192, "by": 12},
                        ],
                    },
                ],
            },
            12,
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
                    "number": list(range(1, 51)),
                },
            ],
            {
                "coords": [[x] for x in range(0, 193, 6)],
            },
            6,
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
                    "number": list(range(0, 51)),
                }
            ],
            {
                "type": "legacywindow",
                "windows": [
                    {
                        "operation": "mean",
                        "grib_keys": {"type": "fcmean"},
                        "deaccumulate": True,
                        "include_start": True,
                        "coords": [
                            {"from": 0, "to": 744, "by": 24},
                            {"from": 744, "to": 1464, "by": 24},
                            {"from": 1464, "to": 2208, "by": 24},
                        ],
                    },
                    {
                        "operation": "standard_deviation",
                        "grib_keys": {"type": "fcstdev"},
                        "deaccumulate": True,
                        "include_start": True,
                        "coords": [
                            {"from": 0, "to": 744, "by": 24},
                            {"from": 744, "to": 1464, "by": 24},
                            {"from": 1464, "to": 2208, "by": 24},
                        ],
                    },
                    {
                        "operation": "maximum",
                        "grib_keys": {"type": "fcmax"},
                        "deaccumulate": True,
                        "include_start": True,
                        "coords": [
                            {"from": 0, "to": 744, "by": 24},
                            {"from": 744, "to": 1464, "by": 24},
                            {"from": 1464, "to": 2208, "by": 24},
                        ],
                    },
                ],
            },
            24,
        ],
    ],
    ids=["accumulate", "ensms", "monthly-stats"],
)
def test_from_inputs(request, entrypoint, input_request, step_accum, stepby):
    test_schema = Schema(schema())

    overrides, cfg_type, updates = TEST_CASES[request.node.callspec.id]
    config = ConfigFactory.from_inputs(
        test_schema, entrypoint, input_request, **overrides
    )
    assert type(config) == cfg_type
    param = input_request[0]["param"]
    check = default_config(param)
    updates["parameters"][param]["accumulations"]["step"] = step_accum
    deep_update(check, updates)
    assert config.model_dump(exclude_defaults=True, by_alias=True) == cfg_type(
        **check
    ).model_dump(exclude_defaults=True, by_alias=True)
    check_request_equality(
        list(config.in_mars()),
        [
            {
                "source": "fdb",
                **req,
                "step": list(range(req["step"][0], req["step"][-1] + 1, stepby)),
            }
            for req in input_request
        ],
    )


def test_wind():
    test_schema = Schema(schema())

    input_request = [
        {
            **DEFAULT_REQUEST,
            "levelist": [250, 500],
            "step": list(range(0, 193, 6)),
            "type": "cf",
            "param": [165, 166],
        },
        {
            **DEFAULT_REQUEST,
            "levelist": [250, 500],
            "step": list(range(0, 193, 6)),
            "type": "pf",
            "param": [165, 166],
            "number": list(range(1, 51)),
        },
    ]
    config = ConfigFactory.from_inputs(test_schema, "pproc-ensms", input_request)
    check_request_equality(
        list(config.out_mars()),
        sum(
            [
                [
                    {
                        "target": "fdb",
                        **DEFAULT_REQUEST,
                        "levelist": [250, 500],
                        "step": list(range(0, 193, 6)),
                        "type": tp,
                        "param": param,
                    }
                    for tp in ["em", "es"]
                ]
                for param in [207, 165, 166]
            ],
            [],
        ),
    )
