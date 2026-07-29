# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime, timedelta

import pytest
from earthkit.data.testing import NO_MARS

from ppruntime.io import retrieve

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(TEST_DIR, "data")

request = {
    "class": "od",
    "expver": "0001",
    "stream": "oper",
    "type": "fc",
    "date": (datetime.today() - timedelta(days=1)).strftime("%Y%m%d"),
    "time": "12",
    "domain": "g",
    "levtype": "sfc",
    "step": "12",
    "param": 228,
}


@pytest.mark.skipif(NO_MARS, reason="No access to MARS")
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"interpolate": {"grid": "O640"}},
        {
            "param": [138, 155],
            "levtype": "pl",
            "levelist": [250, 850],
            "interpolate": {"grid": "O640", "vod2uv": True},
        },
    ],
    ids=["default", "interpolate", "wind"],
)
def test_mars_retrieve(overrides):
    test_request = request.copy()
    test_request.update(overrides)
    retrieve([{"name": "mars"}], [test_request])


@pytest.mark.parametrize(
    "stream",
    [True, False],
)
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"interpolate": {"grid": "O640"}},
        {
            "date": "20240507",
            "stream": "enfo",
            "type": "cf",
            "step": 3,
            "param": [138, 155],
            "levtype": "pl",
            "levelist": [250, 850],
            "interpolate": {"grid": "O640", "vod2uv": True},
        },
    ],
    ids=["default", "interpolate", "wind"],
)
def test_fdb_retrieve(fdb, stream, overrides):
    test_request = {
        "class": "od",
        "expver": "0001",
        "stream": "enfo",
        "type": "cf",
        "date": "20240507",
        "time": "12",
        "domain": "g",
        "levtype": "sfc",
        "step": "12",
        "param": 167,
    }
    test_request.update(overrides)
    retrieve([{"name": "fdb", "stream": stream}], [test_request])


@pytest.mark.parametrize(
    "source",
    [
        {"name": "file", "path": os.path.join(DATA_DIR, "test_2t_12.grib")},
        {
            "name": "file-pattern",
            "pattern": os.path.join(DATA_DIR, "test_{param}_{step}.grib"),
            "hive_partitioning": True,
        },
    ],
    ids=["file", "file-pattern"],
)
def test_file_retrieve(source):
    request = {
        "stream": "enfo",
        "type": "cf",
        "date": "20240507",
        "time": "12",
        "param": "2t",
        "expver": "0001",
        "levtype": "sfc",
    }
    retrieve([source], [request])


def test_multi_source_retrieve(fdb):
    request = {
        "stream": "enfo",
        "type": "cf",
        "date": "20240507",
        "time": "12",
        "param": "2t",
        "expver": "0001",
        "levtype": "sfc",
    }
    sources = [
        {"name": "fdb", "stream": True},
        {
            "name": "file",
            "path": os.path.join(DATA_DIR, "test_2t_12.grib"),
        },
    ]
    retrieve(sources, [request])
