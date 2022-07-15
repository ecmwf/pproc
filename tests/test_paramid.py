# coding: utf-8

import pytest
from pproc.Config import ParamId


METKIT_SHARE_DIR = "~/git/metkit/share/metkit"

pid = ParamId(METKIT_SHARE_DIR)


TESTS_PID = (
    ("12.128", 12),
    (13.128, 13),
    ("14.129", 129014),
    (15.129, 129015),
    (16.1, 100016),
    ("t", 130),
)

TESTS_WIND = (
    (("vo", "d"), ("u", "v")),
    ((138, 155), (131, 132)),
)


@pytest.mark.parametrize("test", TESTS_PID)
def test_paramid(test):
    pin, pout = test
    assert pout == pid.id(pin)


@pytest.mark.parametrize("test", TESTS_WIND)
def test_wind(test):
    vo, d = map(pid.id, test[0])
    u, v = map(pid.id, test[1])
    assert u, v == pid.uv((vo, d))
    assert vo, d == pid.vod((u, v))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test paramId")
    parser.add_argument(
        "--metkit-share-dir",
        help="Metkit configuration directory",
        default="~/git/metkit/share/metkit",
    )

    args = parser.parse_args()

    for fun, tests in zip([test_paramid, test_wind], [TESTS_PID, TESTS_WIND]):
        for test in tests:
            try:
                fun(test)
            except Exception as e:
                print(e)
