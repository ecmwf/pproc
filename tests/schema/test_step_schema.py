# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from pproc.schema.step import StepSchema

from conftest import schema


@pytest.mark.parametrize(
    "out, expected, in_steps",
    [
        [
            {"stream": "enfo", "type": "em", "param": "167", "time": "00"},
            list(range(0, 145, 3)) + list(range(150, 361, 6)),
            list(range(0, 91)) + list(range(93, 145, 3)) + list(range(150, 361, 6)),
        ],
        [
            {"stream": "enfo", "type": "em", "param": "167", "time": "00"},
            list(range(0, 361, 12)),
            list(range(0, 361, 12)),
        ],
        [
            {"stream": "enfo", "type": "cf", "param": "261001", "time": "00"},
            list(range(1, 91)) + list(range(93, 145, 3)) + list(range(150, 361, 6)),
            list(range(0, 91)) + list(range(93, 145, 3)) + list(range(150, 361, 6)),
        ],
        [
            {"stream": "eefo", "type": "fcmean", "param": "167", "time": "00"},
            [f"{x}-{x+168}" for x in list(range(0, 1104 - 168 + 1, 24))],
            list(range(0, 1105, 6)),
        ],
        [
            {
                "stream": "msmm",
                "type": "fcmean",
                "param": "167",
                "date": "20241001",
                "time": "00",
            },
            list(range(1, 8)),
            list(range(0, 5160, 6)),
        ],
        [
            {"stream": "enfo", "type": "ep", "param": "131064", "time": "00"},
            ["120-240", "240-360", "120-168", "168-240"],
            list(range(0, 91)) + list(range(93, 145, 3)) + list(range(150, 361, 6)),
        ],
        [
            {"stream": "oper", "type": "fc", "param": "207", "time": "00"},
            list(range(1, 91)),
            list(range(1, 91)),
        ],
    ],
    ids=[
        "ensms_default",
        "ensms_insteps",
        "deaccumulate",
        "weekly",
        "monthly",
        "prob",
        "fc",
    ],
)
def test_out_steps(out, expected, in_steps):
    test_schema = StepSchema(schema("windows"))
    _, out_steps = test_schema.out_steps(out, in_steps)
    assert out_steps == expected
