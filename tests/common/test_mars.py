from datetime import date, datetime

import pytest

from pproc.common import mars


@pytest.mark.parametrize(
    "val, expected",
    [
        pytest.param("abc", b"abc", id="str"),
        pytest.param(b"def", b"def", id="bytes"),
        pytest.param(123, b"123", id="int"),
        pytest.param(date(2021, 12, 13), b"20211213", id="date"),
        pytest.param(datetime(2022, 5, 8, 14, 36, 25), b"20220508", id="datetime"),
        pytest.param(range(5), b"0/to/4", id="range1"),
        pytest.param(range(2, 8), b"2/to/7", id="range2"),
        pytest.param(range(1, 9, 3), b"1/to/7/by/3", id="range3"),
        pytest.param(("u", "v", "z"), b"u/v/z", id="tuple"),
        pytest.param([1, 3, 4, 9], b"1/3/4/9", id="list"),
    ],
)
def test_val_to_mars(val, expected):
    assert mars._val_to_mars(val) == expected


def test_to_mars():
    req = {
        "class": "od",
        "expver": "0001",
        "stream": "enfo",
        "date": datetime(2020, 10, 11),
        "time": "0000",
        "type": "pf",
        "number": range(1, 51),
        "step": [24, 36, 72],
        "levtype": "pl",
        "levelist": 500,
        "param": (129, 130),
    }
    expected = b"retrieve,class=od,expver=0001,stream=enfo,date=20201011,time=0000,type=pf,number=1/to/50,step=24/36/72,levtype=pl,levelist=500,param=129/130"
    assert mars.to_mars(b"retrieve", req) == expected
