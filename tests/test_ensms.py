# coding: utf-8

import pytest
from pproc.Config import VariableTree
from os import path

tree = VariableTree(path.join(path.dirname(__file__), "test_ensms.yaml"))

TESTS_VARS = (
    (
        ("ensms", 48, "t"),
        {
            "step": "0/to/48/by/3",
            "param": 130,
            "levtype": "pl",
            "levelist": "250/500/850",
        },
    ),
    (
        ("ensms", 96, "mx2t6"),
        {"step": "54/to/96/by/6", "param": 121, "levtype": "sfc", "levelist": None},
    ),
    (
        ("ensms", 360, "mn2t6"),
        {"step": "306/to/360/by/6", "param": 122, "levtype": "sfc", "levelist": None},
    ),
)


@pytest.mark.parametrize("test", TESTS_VARS)
def test_variables(test):
    path, variables = test
    x = tree.variables(*path)
    print(x)
    print(variables)
    assert x == variables


if __name__ == "__main__":
    for test in TESTS_VARS:
        try:
            test_variables(test)
        except Exception as e:
            print(e)
