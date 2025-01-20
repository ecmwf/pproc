import pytest
import os

from pproc.config.schema import Schema

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "req, config",
    [
        [
            {
                "class": "od",
                "stream": "enfo",
                "expver": "0001",
                "levtype": "pl",
                "levelist": [250, 500],
                "domain": "g",
                "param": "167.128",
                "date": "20241001",
                "time": "0",
                "step": "12-744",
                "type": "fcmean",
            },
            {
                "entrypoint": "pproc-accumulate",
                "request": {
                    "class": "od",
                    "stream": "enfo",
                    "expver": "0001",
                    "levtype": "pl",
                    "levelist": [250, 500],
                    "domain": "g",
                    "param": "167.128",
                    "date": "20241001",
                    "time": "0",
                    "type": ["cf", "pf"],
                },
                "members": 50,
                "total_fields": 51,
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "operation": "mean",
                        "grib_keys": {"type": "fcmean"},
                        "coords": [
                            {
                                "from": "12",
                                "to": "744",
                                "by": "6",
                            }
                        ],
                    }
                },
            },
        ],
        [
            {
                "class": "od",
                "stream": "enfo",
                "expver": "0001",
                "levtype": "sfc",
                "domain": "g",
                "param": "228.172",
                "date": "20241001",
                "time": "00",
                "step": "12-744",
                "type": "fcmean",
            },
            {
                "entrypoint": "pproc-accumulate",
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "operation": "mean",
                        "deaccumulate": True,
                        "include_start_step": True,
                        "grib_keys": {"type": "fcmean"},
                        "coords": [
                            {
                                "from": "12",
                                "to": "744",
                                "by": "24",
                            }
                        ],
                    }
                },
                "members": 50,
                "total_fields": 51,
                "vmin": 0.0,
                "preprocessing": [
                    {
                        "operation": "scale",
                        "value": 0.00001157407,
                    }
                ],
                "request": {
                    "class": "od",
                    "stream": "enfo",
                    "expver": "0001",
                    "levtype": "sfc",
                    "domain": "g",
                    "param": "228.128",
                    "date": "20241001",
                    "time": "00",
                    "type": ["cf", "pf"],
                },
                "metadata": {
                    "paramId": 172228,
                },
            },
        ],
    ],
    ids=["2t", "tp"],
)
def test_schema_from_output(req, config):
    schema = Schema(os.path.join(TEST_DIR, "schema.yaml"))
    assert config == schema.config_from_output(req)


@pytest.mark.parametrize(
    "entrypoint, req, num_expected, expected",
    [
        [
            "pproc-accumulate",
            {
                "class": "od",
                "stream": "enfo",
                "expver": "0001",
                "levtype": "sfc",
                "domain": "g",
                "param": "167.128",
                "date": "20241001",
                "time": "0",
                "step": [0, 6, 12, 18, 24],
                "type": ["cf", "pf"],
            },
            4,
            {
                "entrypoint": "pproc-accumulate",
                "request": {
                    "class": "od",
                    "stream": "enfo",
                    "expver": "0001",
                    "levtype": "sfc",
                    "domain": "g",
                    "param": "167.128",
                    "date": "20241001",
                    "time": "0",
                    "type": ["cf", "pf"],
                },
                "members": 50,
                "total_fields": 51,
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "operation": "mean",
                        "grib_keys": {"type": "fcmean"},
                        "coords": {
                            "type": "ranges",
                            "from": "0",
                            "to": "24",
                            "by": "6",
                            "interval": 24,
                            "width": 168,
                        },
                    }
                },
            },
        ],
        [
            "pproc-monthly-stats",
            {
                "class": "od",
                "stream": "mmsf",
                "expver": "0001",
                "levtype": "sfc",
                "domain": "g",
                "param": "228.128",
                "date": "20241001",
                "time": "00",
                "step": list(range(24, 6997, 6)),
                "number": list(range(1, 21)),
                "type": "fc",
            },
            3,
            {
                "entrypoint": "pproc-monthly-stats",
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "operation": "mean",
                        "deaccumulate": True,
                        "include_start_step": True,
                        "grib_keys": {"type": "fcmean"},
                        "coords": {
                            "type": "monthly",
                            "from": "24",
                            "to": "6996",
                            "by": "6",
                            "date": "20241001",
                        },
                    }
                },
                "members": {"start": 1, "end": 20},
                "vmin": 0.0,
                "preprocessing": [
                    {
                        "operation": "scale",
                        "value": 0.00001157407,
                    }
                ],
                "request": {
                    "class": "od",
                    "stream": "mmsf",
                    "expver": "0001",
                    "levtype": "sfc",
                    "domain": "g",
                    "param": "228.128",
                    "date": "20241001",
                    "time": "00",
                    "type": "fc",
                },
                "metadata": {
                    "paramId": 172228,
                    "stream": "msmm",
                },
            },
        ],
    ],
    ids=["2t", "tp"],
)
def test_schema_from_input(entrypoint, req, num_expected, expected):
    schema = Schema(os.path.join(TEST_DIR, "schema.yaml"))
    configs = list(schema.config_from_input(entrypoint, req))
    assert len(configs) == num_expected
    assert configs[0] == expected
