# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
from conftest import schema

from ppcore.schema.schema import Schema


@pytest.mark.parametrize(
    "req, config, num_generated",
    [
        [
            {
                "class": "od",
                "stream": "msmm",
                "expver": "0001",
                "levtype": "sfc",
                "domain": "g",
                "param": 167,
                "date": "20241001",
                "time": "0",
                "fcmonth": 1,
                "type": "fcmean",
            },
            {
                "entrypoint": "pproc-monthly-stats",
                "name": "167_sfc",
                "interp_keys": {
                    "intgrid": "none",
                    "legendre-loader": "shmem",
                    "matrix-loader": "file-io",
                },
                "dtype": "float64",
                "inputs": [
                    {
                        "class": "od",
                        "stream": "mmsf",
                        "expver": "0001",
                        "levtype": "sfc",
                        "domain": "g",
                        "param": "167",
                        "date": "20241001",
                        "time": "0000",
                        "type": "fc",
                        "step": list(range(6, 745, 6)),
                        "number": list(range(0, 51)),
                    },
                ],
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "operation": "mean",
                        "metadata": {"type": "fcmean", "bitsPerValue": 16},
                        "name": {
                            "type": "monthly",
                            "date": "20241001",
                        },
                    }
                },
            },
            1,
        ],
        [
            {
                "class": "od",
                "stream": "eefo",
                "expver": "0001",
                "levtype": "sfc",
                "domain": "g",
                "param": 172228,
                "date": "20241001",
                "time": "00",
                "step": "0-168",
                "type": "fcmean",
                "number": [0, 1, 2, 3],
            },
            {
                "entrypoint": "pproc-accumulate",
                "name": "228_sfc",
                "interp_keys": {
                    "intgrid": "none",
                    "legendre-loader": "shmem",
                    "matrix-loader": "file-io",
                },
                "dtype": "float64",
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "operation": "mean",
                        "deaccumulate": True,
                        "metadata": {
                            "type": "fcmean",
                            "bitsPerValue": 16,
                            "legBaseDate": 0,
                            "legNumber": 0,
                            "numberIncludedInAverage": "{num_coords}:int",
                            "stepType": "avg",
                            "timeRangeIndicator": 3,
                        },
                        "name": {
                            "type": "default",
                            "length": 168,
                        },
                    }
                },
                "vmin": 0.0,
                "preprocessing": [
                    {
                        "operation": "scale",
                        "value": 0.00001157407,
                    }
                ],
                "inputs": [
                    {
                        "class": "od",
                        "stream": "eefo",
                        "expver": "0001",
                        "levtype": "sfc",
                        "domain": "g",
                        "param": "228",
                        "date": "20241001",
                        "time": "0000",
                        "type": "cf",
                        "step": list(range(0, 169, 24)),
                    },
                    {
                        "class": "od",
                        "stream": "eefo",
                        "expver": "0001",
                        "levtype": "sfc",
                        "domain": "g",
                        "param": "228",
                        "date": "20241001",
                        "time": "0000",
                        "type": "pf",
                        "step": list(range(0, 169, 24)),
                        "number": list(range(1, 4)),
                    },
                ],
            },
            2,
        ],
        [
            {
                "class": "od",
                "stream": "enfo",
                "expver": "0001",
                "levtype": "pl",
                "levelist": [250, 850],
                "domain": "g",
                "param": "130",
                "date": "20241001",
                "time": "00",
                "step": 12,
                "type": "em",
                "target_grid": "O640",
            },
            {
                "entrypoint": "pproc-ensms",
                "name": "130_pl",
                "interp_keys": {
                    "intgrid": "none",
                    "legendre-loader": "shmem",
                    "matrix-loader": "file-io",
                },
                "input": {
                    "dtype": "float64",
                },
                "statistics": {
                    "metadata": {
                        "bitsPerValue": 16,
                        "legBaseDate": 20241001,
                        "legBaseTime": 0,
                        "legNumber": 1,
                        "localDefinitionNumber": 30,
                        "numberOfForecastsInEnsemble": "{num_fields}:int",
                        "oceanAtmosphereCoupling": 2,
                        "perturbationNumber": 0,
                    },
                    "operation": "mean",
                },
                "inputs": [
                    {
                        "class": "od",
                        "stream": "oper",
                        "expver": "0001",
                        "levtype": "pl",
                        "domain": "g",
                        "param": "130",
                        "date": "20241001",
                        "time": "0000",
                        "type": "fc",
                        "levelist": [250, 850],
                        "step": 12,
                        "target_grid": "O640",
                    },
                    {
                        "class": "od",
                        "stream": "enfo",
                        "expver": "0001",
                        "levtype": "pl",
                        "domain": "g",
                        "param": "130",
                        "date": "20241001",
                        "time": "0000",
                        "type": "pf",
                        "levelist": [250, 850],
                        "step": 12,
                        "number": list(range(1, 51)),
                        "target_grid": "O640",
                    },
                ],
            },
            1,
        ],
    ],
    ids=["2t", "tp", "T"],
)
def test_schema_from_output(req, config, num_generated):
    test_schema = Schema(**schema())
    test_config = test_schema.config_from_output(req)
    test_config.pop("metadata", None)
    assert config == test_config

    generated = test_schema.config_from_input(
        config["inputs"], {k: req[k] for k in ["stream", "type", "param"]}
    )
    assert len(list(generated)) == num_generated


@pytest.mark.parametrize(
    "entrypoint, req, num_expected, expected",
    [
        [
            "pproc-accumulate",
            [
                {
                    "class": "od",
                    "stream": "enfo",
                    "expver": "0001",
                    "levtype": "sfc",
                    "domain": "g",
                    "param": "167",
                    "date": "20241001",
                    "time": "0",
                    "step": list(range(0, 169, 6)),
                    "type": "pf",
                    "number": list(range(1, 11)),
                },
                {
                    "class": "od",
                    "stream": "oper",
                    "expver": "0001",
                    "levtype": "sfc",
                    "domain": "g",
                    "param": "167",
                    "date": "20241001",
                    "time": "0",
                    "step": list(range(0, 169, 6)),
                    "type": "fc",
                },
            ],
            8,
            {
                "entrypoint": "pproc-accumulate",
                "name": "167_sfc",
                "inputs": [
                    {
                        "class": "od",
                        "stream": "oper",
                        "expver": "0001",
                        "levtype": "sfc",
                        "domain": "g",
                        "param": "167",
                        "date": "20241001",
                        "time": "0000",
                        "type": "fc",
                        "step": list(range(6, 169, 6)),
                    }
                ],
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "operation": "mean",
                        "metadata": {
                            "type": "fcmean",
                            "bitsPerValue": 16,
                            "legBaseDate": 0,
                            "legNumber": 0,
                            "numberIncludedInAverage": "{num_coords}:int",
                            "stepType": "avg",
                            "timeRangeIndicator": 3,
                        },
                        "name": {
                            "type": "default",
                            "length": 168,
                        },
                    }
                },
                "interp_keys": {
                    "intgrid": "none",
                    "legendre-loader": "shmem",
                    "matrix-loader": "file-io",
                },
                "dtype": "float64",
            },
        ],
        [
            "pproc-monthly-stats",
            [
                {
                    "class": "od",
                    "stream": "mmsf",
                    "expver": "0001",
                    "levtype": "sfc",
                    "domain": "g",
                    "param": 228,
                    "date": "20241001",
                    "time": "00",
                    "step": list(range(0, 5161, 6)),
                    "number": list(range(1, 21)),
                    "type": "fc",
                }
            ],
            21,
            {
                "entrypoint": "pproc-monthly-stats",
                "name": "228_sfc",
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "operation": "mean",
                        "deaccumulate": True,
                        "metadata": {
                            "type": "fcmean",
                            "bitsPerValue": 16,
                        },
                        "name": {
                            "type": "monthly",
                            "date": "20241001",
                        },
                    }
                },
                "vmin": 0.0,
                "dtype": "float64",
                "preprocessing": [
                    {
                        "operation": "scale",
                        "value": 0.00001157407,
                    }
                ],
                "inputs": [
                    {
                        "class": "od",
                        "stream": "mmsf",
                        "expver": "0001",
                        "levtype": "sfc",
                        "domain": "g",
                        "param": "228",
                        "date": "20241001",
                        "time": "0000",
                        "type": "fc",
                        "step": list(range(0, 745, 24)),
                        "number": list(range(1, 21)),
                    }
                ],
                "interp_keys": {
                    "intgrid": "none",
                    "legendre-loader": "shmem",
                    "matrix-loader": "file-io",
                },
            },
        ],
        [
            "pproc-ensms",
            [
                {
                    "class": "od",
                    "stream": "enfo",
                    "expver": "0001",
                    "levtype": "pl",
                    "levelist": [250, 850],
                    "domain": "g",
                    "param": "130",
                    "date": "20241001",
                    "time": "0",
                    "step": [0, 6, 12, 18, 24],
                    "type": "pf",
                    "number": list(range(1, 51)),
                },
                {
                    "class": "od",
                    "stream": "oper",
                    "expver": "0001",
                    "levtype": "pl",
                    "levelist": [250, 850],
                    "domain": "g",
                    "param": "130",
                    "date": "20241001",
                    "time": "0",
                    "step": [0, 6, 12, 18, 24],
                    "type": "fc",
                },
            ],
            10,
            {
                "entrypoint": "pproc-ensms",
                "name": "130_pl",
                "inputs": [
                    {
                        "class": "od",
                        "stream": "oper",
                        "expver": "0001",
                        "levtype": "pl",
                        "levelist": [250, 850],
                        "domain": "g",
                        "param": "130",
                        "date": "20241001",
                        "time": "0000",
                        "type": "fc",
                        "step": 0,
                    },
                    {
                        "class": "od",
                        "stream": "enfo",
                        "expver": "0001",
                        "levtype": "pl",
                        "levelist": [250, 850],
                        "domain": "g",
                        "param": "130",
                        "date": "20241001",
                        "time": "0000",
                        "type": "pf",
                        "step": 0,
                        "number": list(range(1, 51)),
                    },
                ],
                "interp_keys": {
                    "intgrid": "none",
                    "legendre-loader": "shmem",
                    "matrix-loader": "file-io",
                },
                "input": {
                    "dtype": "float64",
                },
                "statistics": {
                    "metadata": {
                        "bitsPerValue": 16,
                        "legBaseDate": 20241001,
                        "legBaseTime": 0,
                        "legNumber": 1,
                        "localDefinitionNumber": 30,
                        "numberOfForecastsInEnsemble": "{num_fields}:int",
                        "oceanAtmosphereCoupling": 2,
                        "perturbationNumber": 0,
                    },
                    "operation": "mean",
                },
            },
        ],
    ],
    ids=["2t", "tp", "T"],
)
def test_schema_from_input(entrypoint, req, num_expected, expected):
    test_schema = Schema(**schema())
    configs = list(test_schema.config_from_input(req, entrypoint=entrypoint))
    assert len(configs) == num_expected
    test_config = configs[0]
    test_config.pop("metadata", None)
    assert test_config == expected
