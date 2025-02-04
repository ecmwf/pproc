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
                "param": 167,
                "date": "20241001",
                "time": "0",
                "step": "12-744",
                "type": "fcmean",
            },
            {
                "entrypoint": "pproc-accumulate",
                "request": [
                    {
                        "class": "od",
                        "stream": "enfo",
                        "expver": "0001",
                        "levtype": "pl",
                        "levelist": [250, 500],
                        "domain": "g",
                        "param": "167",
                        "date": "20241001",
                        "time": "0",
                        "type": "cf",
                    },
                    {
                        "class": "od",
                        "stream": "enfo",
                        "expver": "0001",
                        "levtype": "pl",
                        "levelist": [250, 500],
                        "domain": "g",
                        "param": "167",
                        "date": "20241001",
                        "time": "0",
                        "type": "pf",
                    },
                ],
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
                "param": 172228,
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
                "request": [
                    {
                        "class": "od",
                        "stream": "enfo",
                        "expver": "0001",
                        "levtype": "sfc",
                        "domain": "g",
                        "param": "228",
                        "date": "20241001",
                        "time": "00",
                        "type": "cf",
                    },
                    {
                        "class": "od",
                        "stream": "enfo",
                        "expver": "0001",
                        "levtype": "sfc",
                        "domain": "g",
                        "param": "228",
                        "date": "20241001",
                        "time": "00",
                        "type": "pf",
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
                "step": [0, 6, 12],
                "type": "em",
                "interp_grid": "O640",
            },
            {
                "entrypoint": "pproc-ensms",
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "coords": [[0], [6], [12]],
                    },
                },
                "members": 50,
                "total_fields": 51,
                "request": [
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
                        "interpolate": {
                            "grid": "O640",
                            "intgrid": "none",
                            "legendre-loader": "shmem",
                            "matrix-loader": "file-io",
                        },
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
                        "interpolate": {
                            "grid": "O640",
                            "intgrid": "none",
                            "legendre-loader": "shmem",
                            "matrix-loader": "file-io",
                        },
                    },
                ],
                "metadata": {
                    "bitsPerValue": 16,
                    "perturbationNumber": 0,
                },
            },
        ],
    ],
    ids=["2t", "tp", "T"],
)
def test_schema_from_output(req, config):
    schema = Schema(os.path.join(TEST_DIR, "schema.yaml"))
    assert config == schema.config_from_output(req)


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
                    "param": 167,
                    "date": "20241001",
                    "time": "0",
                    "step": [0, 6, 12, 18, 24],
                    "type": "pf",
                },
                {
                    "class": "od",
                    "stream": "enfo",
                    "expver": "0001",
                    "levtype": "sfc",
                    "domain": "g",
                    "param": 167,
                    "date": "20241001",
                    "time": "0",
                    "step": [0, 6, 12, 18, 24],
                    "type": "cf",
                },
            ],
            4,
            {
                "entrypoint": "pproc-accumulate",
                "request": [
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
                    },
                ],
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
                    "step": list(range(24, 6997, 6)),
                    "number": list(range(1, 21)),
                    "type": "fc",
                }
            ],
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
                "request": [
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
                    }
                ],
                "metadata": {
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
                    "interp_grid": "O640",
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
                    "interp_grid": "O640",
                },
            ],
            2,
            {
                "entrypoint": "pproc-ensms",
                "request": [
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
                        "interpolate": {
                            "grid": "O640",
                            "intgrid": "none",
                            "legendre-loader": "shmem",
                            "matrix-loader": "file-io",
                        },
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
                        "interpolate": {
                            "grid": "O640",
                            "intgrid": "none",
                            "legendre-loader": "shmem",
                            "matrix-loader": "file-io",
                        },
                    },
                ],
                "members": 50,
                "total_fields": 51,
                "accumulations": {
                    "step": {
                        "type": "legacywindow",
                        "coords": {
                            "type": "ranges",
                            "from": "0",
                            "to": "24",
                            "interval": "6",
                        },
                    }
                },
                "metadata": {
                    "bitsPerValue": 16,
                    "perturbationNumber": 0,
                },
            },
        ],
    ],
    ids=["2t", "tp", "T"],
)
def test_schema_from_input(entrypoint, req, num_expected, expected):
    schema = Schema(os.path.join(TEST_DIR, "schema.yaml"))
    configs = list(schema.config_from_input(req, entrypoint=entrypoint))
    assert len(configs) == num_expected
    assert configs[0] == expected
