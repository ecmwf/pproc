# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from datetime import datetime, timedelta

import pytest

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


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"interpolate": {"grid": "O640"}},
        {
            "param": [138, 155],
            "levtype": "pl",
            "levelist": [250, 850],
            "interpolate": {"grid": "O640", "vod2uv": "1"},
        },
    ],
    ids=["default", "interpolate", "wind"],
)
def test_mars_retrieve(overrides):
    test_request = request.copy()
    test_request.update(overrides)
    retrieve(["mars"], [test_request])


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
            "param": [138, 155],
            "levtype": "pl",
            "levelist": [250, 850],
            "interpolate": {"grid": "O640", "vod2uv": "1"},
        },
    ],
    ids=["default", "interpolate", "wind"],
)
def test_fdb_retrieve(stream, overrides):
    os.environ["FDB_HOME"] = "/home/fdbprod"
    test_request = request.copy()
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


def test_multi_source_retrieve():
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
    os.environ["FDB_HOME"] = "/home/fdbprod"
    retrieve(sources, [request])
