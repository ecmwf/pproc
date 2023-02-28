
from typing import Dict, Tuple

import pytest

from pproc.common.resources import TimeDecomposition, plural, pretty_bytes


def is_close(x, y, eps=1e-12):
    if y == 0:
        return abs(x) < eps
    return abs(x - y) / y < eps


def test_plural():
    assert plural(-32, "bar") == "bars"
    assert plural(0, "egg") == "egg"
    assert plural(1, "foo") == "foo"
    assert plural(2, "bean") == "beans"
    assert plural(153, "spam") == "spams"
    assert plural(-12.6, "mile") == "miles"
    assert plural(1.5, "day") == "day"
    assert plural(2.3, "tree") == "trees"
    assert plural(18.95e8, "electron") == "electrons"


@pytest.mark.parametrize("inp, exp", [
    ({}, (0, 0, 0, 0, 0.)),
    ({"days": 12.5}, (12, 12, 0, 0, 0.)),
    ({"seconds": 124.008}, (0, 0, 2, 4, 8000.)),
    ({"days": 1, "hours": 23, "minutes": 59, "seconds": 59, "microseconds": 1e6}, (2, 0, 0, 0, 0.)),
])
def test_time_decomp(inp: Dict[str, float], exp: Tuple[int, int, int, int, float]):
    d, h, m, s, u = exp
    t = TimeDecomposition(**inp)
    assert t.days == d
    assert t.hours == h
    assert t.minutes == m
    assert t.seconds == s
    assert is_close(t.microseconds, u)


@pytest.mark.parametrize("inp, exp", [
    ({}, 0.),
    ({"days": 1}, 86400.),
    ({"hours": 1.5}, 5400.),
    ({"minutes": 72}, 4320.),
    ({"seconds": 123456.789}, 123456.789),
    ({"microseconds": 3e5}, 0.3),
])
def test_timedecomp_seconds(inp: Dict[str, float], exp: float):
    t = TimeDecomposition(**inp)
    assert t.total_seconds() == exp


@pytest.mark.parametrize("inp, exp", [
    ({"days": 3.25}, "280800 s (3 days 6 hours)"),
    ({"days": 1, "hours": 1.2}, "90720 s (1 day 1 hour)"),
    ({"minutes": 84}, "5040 s (1 hour 24 minutes)"),
    ({"seconds": 3600}, "3600 s (1 hour)"),
    ({"seconds": 3.5}, "3.5 s (3 seconds 500000 microseconds)"),
    ({"microseconds": 1}, "1e-06 s (1 microsecond)"),
    ({}, "0 s"),
])
def test_timedecomp_pretty(inp: Dict[str, float], exp: str):
    t = TimeDecomposition(**inp)
    assert t.pretty() == exp


@pytest.mark.parametrize("inp, exp", [
    (1, "1 byte"),
    (5, "5 bytes"),
    (1024, "1024 bytes (1 kiB)"),
    (6081740, "6081740 bytes (5.8 MiB)"),
    (38654705664, "38654705664 bytes (36 GiB)"),
    (21220574416077, "21220574416077 bytes (19.3 TiB)"),
    (88101667710000000, "88101667710000000 bytes (78.25 PiB)"),
    (1152921504606846976, "1152921504606846976 bytes (1 EiB)"),
])
def test_pretty_bytes(inp: int, exp: str):
    assert pretty_bytes(inp) == exp


@pytest.mark.parametrize("inp, exp", [
    (1, "1 byte"),
    (3, "3 bytes"),
    (1024, "1024 bytes (1.024 kB)"),
    (57890912, "57890912 bytes (57.8909 MB)"),
    (453468743148, "453468743148 bytes (453.469 GB)"),
    (1546431843514, "1546431843514 bytes (1.54643 TB)"),
    (46546843548643475, "46546843548643475 bytes (46.5468 PB)"),
    (1152921504606846976, "1152921504606846976 bytes (1.15292 EB)"),
])
def test_pretty_bytes_dec(inp: int, exp: str):
    assert pretty_bytes(inp, decimal=True) == exp