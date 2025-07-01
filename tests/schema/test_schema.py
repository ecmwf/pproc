# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import os

from pproc.schema.schema import Schema

from conftest import schema


@pytest.mark.parametrize(
    "req, config",
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
                        "time": "0",
                        "type": "fc",
                        "step": list(range(0, 745, 6)),
                        "number": list(range(0, 51)),
                    },
                ],
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "operation": "mean",
                        "metadata": {"type": "fcmean"},
                    }
                },
                "metadata": {
                    "bitsPerValue": 16,
                    "stream": "msmm",
                },
            },
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
                        "include_start": True,
                        "metadata": {"type": "fcmean"},
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
                        "time": "00",
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
                        "time": "00",
                        "type": "pf",
                        "step": list(range(0, 169, 24)),
                        "number": list(range(1, 101)),
                    },
                ],
                "metadata": {
                    "paramId": 172228,
                },
            },
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
                "dtype": "float64",
                "inputs": [
                    {
                        "class": "od",
                        "stream": "enfo",
                        "expver": "0001",
                        "levtype": "pl",
                        "domain": "g",
                        "param": "130",
                        "date": "20241001",
                        "time": "00",
                        "type": "cf",
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
                        "time": "00",
                        "type": "pf",
                        "levelist": [250, 850],
                        "step": 12,
                        "number": list(range(1, 51)),
                        "target_grid": "O640",
                    },
                ],
                "metadata": {
                    "bitsPerValue": 16,
                    "numberOfForecastsInEnsemble": 51,
                    "perturbationNumber": 0,
                },
            },
        ],
    ],
    ids=["2t", "tp", "T"],
)
def test_schema_from_output(req, config):
    test_schema = Schema(**schema())
    assert config == test_schema.config_from_output(req)

    generated = test_schema.config_from_input(
        config["inputs"], {k: req[k] for k in ["stream", "type", "param"]}
    )
    assert len(list(generated)) == 1


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
                    "stream": "enfo",
                    "expver": "0001",
                    "levtype": "sfc",
                    "domain": "g",
                    "param": "167",
                    "date": "20241001",
                    "time": "0",
                    "step": list(range(0, 169, 6)),
                    "type": "cf",
                },
            ],
            4,
            {
                "entrypoint": "pproc-accumulate",
                "name": "167_sfc",
                "inputs": [
                    {
                        "class": "od",
                        "stream": "enfo",
                        "expver": "0001",
                        "levtype": "sfc",
                        "domain": "g",
                        "param": "167",
                        "date": "20241001",
                        "time": "0",
                        "type": "cf",
                        "step": list(range(0, 169, 6)),
                    },
                    {
                        "class": "od",
                        "stream": "enfo",
                        "expver": "0001",
                        "levtype": "sfc",
                        "domain": "g",
                        "param": "167",
                        "date": "20241001",
                        "time": "0",
                        "type": "pf",
                        "step": list(range(0, 169, 6)),
                        "number": list(range(1, 11)),
                    },
                ],
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "operation": "mean",
                        "metadata": {"type": "fcmean"},
                    }
                },
                "interp_keys": {
                    "intgrid": "none",
                    "legendre-loader": "shmem",
                    "matrix-loader": "file-io",
                },
                "dtype": "float64",
                "metadata": {},
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
                        "include_start": True,
                        "metadata": {"type": "fcmean"},
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
                        "time": "00",
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
                "metadata": {
                    "bitsPerValue": 16,
                    "paramId": 172228,
                    "stream": "msmm",
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
                    "stream": "enfo",
                    "expver": "0001",
                    "levtype": "pl",
                    "levelist": [250, 850],
                    "domain": "g",
                    "param": "130",
                    "date": "20241001",
                    "time": "0",
                    "step": [0, 6, 12, 18, 24],
                    "type": "cf",
                },
            ],
            10,
            {
                "entrypoint": "pproc-ensms",
                "name": "130_pl",
                "inputs": [
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
                        "type": "cf",
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
                        "time": "0",
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
                "dtype": "float64",
                "metadata": {
                    "bitsPerValue": 16,
                    "numberOfForecastsInEnsemble": 51,
                    "perturbationNumber": 0,
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
    assert configs[0] == expected
