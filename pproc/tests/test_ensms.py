# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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
