import datetime
from typing import Iterator

from earthkit.time.calendar import MonthInYear
from earthkit.time.sequence import MonthlySequence


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
