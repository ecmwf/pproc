# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from typing import Iterator

from earthkit.time.calendar import MonthInYear
from earthkit.time.sequence import MonthlySequence


def steprange_to_fcmonth(fcdate: datetime.datetime | str, steprange: str) -> int:
    if isinstance(fcdate, str):
        fcdate = datetime.datetime.strptime(str(fcdate), "%Y%m%d")
    start, end = map(int, steprange.split("-"))
    seq = MonthlySequence(1)
    first_month = seq.next(fcdate, False)
    this_month = fcdate + datetime.timedelta(hours=int(start))

    month_length = MonthInYear(this_month.year, this_month.month).length() * 24
    assert month_length == (
        end - start
    ), f"Expected month length {end - start} for {fcdate} and step range {steprange}. Got {month_length}."

    return (
        (this_month.year - first_month.year) * 12
        + this_month.month
        - first_month.month
        + 1
    )


def fcmonth_to_steprange(fcdate: datetime.datetime | str, fcmonth: int) -> str:
    if isinstance(fcdate, str):
        fcdate = datetime.datetime.strptime(str(fcdate), "%Y%m%d")
    seq = MonthlySequence(1)
    month = seq.next(fcdate, False)
    for _ in range(1, fcmonth):
        month = seq.next(month, True)
    date = datetime.datetime(month.year, month.month, month.day)
    start = (date - fcdate).total_seconds() // 3600
    end = start + MonthInYear(date.year, date.month).length() * 24
    return f"{int(start)}-{int(end)}"


def stepseq_monthly(date: str, start: int, end: int, interval: int) -> Iterator:
    fcdate = datetime.datetime.strptime(str(date), "%Y%m%d")
    start_date = fcdate + datetime.timedelta(hours=start)
    seq = MonthlySequence(1)
    start_month = seq.next(start_date.date(), strict=False)
    # Steps are in hours, from forecast date
    step_start = (start_month - fcdate.date()).days * 24
    miny = MonthInYear(start_month.year, start_month.month)
    while step_start < end:
        delta = miny.length() * 24
        step_end = step_start + delta

        if step_end > end:
            break

        yield range(step_start, step_end + 1, interval)
        miny = miny.next()
        step_start = step_end


def stepseq_ranges(
    start: int, end: int, width: int, interval: int, by: int
) -> Iterator:
    for start in range(start, end - width + 1, interval):
        yield range(start, start + width + 1, by)
