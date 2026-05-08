# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from datetime import datetime

from pproc.common import stepseq


@pytest.mark.parametrize(
    "date, start, end, interval, num_months, remaining",
    [
        ["20241001", 12, 5160, 12, 7, 6],
        ["20241001", 6, 5160, 6, 7, 12],
        ["20241001", 0, 5160, 6, 7, 12],
        ["20241002", 6, 5160, 6, 6, 16],
        ["20241002", 6, 800, 6, 0, 0],
    ],
)
def test_monthly(date, start, end, interval, num_months, remaining):
    steps = list(range(start, end + 1, interval))
    step_ranges = [x for x in stepseq.stepseq_monthly(date, start, end, interval)]
    assert len(step_ranges) == num_months
    if num_months > 0:
        assert len(steps[steps.index(step_ranges[-1][-1]) + 1 :]) == remaining


@pytest.mark.parametrize(
    "start, end, width, interval, by, expected",
    [
        [0, 0, 0, 1, 1, [range(0, 1)]],
        [0, 120, 0, 24, 1, [range(x, x + 1) for x in range(0, 121, 24)]],
        [
            0,
            360,
            120,
            120,
            6,
            [range(0, 121, 6), range(120, 241, 6), range(240, 361, 6)],
        ],
    ],
)
def test_ranges(start, end, width, interval, by, expected):
    assert [
        x for x in stepseq.stepseq_ranges(start, end, width, interval, by)
    ] == expected


@pytest.mark.parametrize(
    "date, fcmonth, expected",
    [
        ["20241001", 1, f"0-{31*24}"],
        ["20241001", 7, f"{182*24}-{(182+30)*24}"],
        ["20241002", 1, f"{30*24}-{(30+30)*24}"],
    ],
)
def test_conversions(date, fcmonth, expected):
    assert (
        stepseq.fcmonth_to_steprange(datetime.strptime(date, "%Y%m%d"), fcmonth)
        == expected
    )
    assert (
        stepseq.steprange_to_fcmonth(datetime.strptime(date, "%Y%m%d"), expected)
        == fcmonth
    )
