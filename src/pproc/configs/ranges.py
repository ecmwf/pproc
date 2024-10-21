from datetime import datetime
from typing import List

import numpy as np


def _increment_month(date: datetime) -> datetime:
    replace = {"day": 1}
    next_month = date.month + 1
    if next_month > 12:
        next_month = next_month % 12
        replace["year"] = date.year + 1
    replace["month"] = next_month
    return date.replace(**replace)


def monthly(date: str, steps: List[int]) -> List[List[int]]:
    """
    Compute list of steps which belong to a specific month,
    starting from a given date and a list of forecast steps.
    The interval between forecast steps must be constant.
    """
    intervals = np.diff(np.array(steps))
    assert np.all(intervals == intervals[0]), "Step intervals must be constant"
    interval = intervals[0]
    start_month = datetime.strptime(date, "%Y%m%d")
    step_ranges = []
    step_index = 0 if steps[0] != 0 else 1
    while step_index < len(steps):
        next_month = _increment_month(start_month)
        delta = int((next_month - start_month).total_seconds() / (60 * 60 * interval))
        month_range = steps[step_index: step_index + delta]
        if start_month.day == 1 and len(month_range) == delta:
            # Only append range if we have the full months
            step_ranges.append(month_range)
        start_month = next_month
        step_index += delta
    return step_ranges
