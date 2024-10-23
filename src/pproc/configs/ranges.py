from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np


def populate_accums(accums: dict, request: dict) -> dict:
    for dim, dim_config in accums.items():
        if dim == "step" and dim_config.get("range_type"):
            accum_type = dim_config["range_type"]
            known = {
                "monthly": monthly,
                "weekly": weekly,
            }
            if accum_type not in known:
                raise ValueError(
                    f"Unknown range type {accum_type}. Must be one of {list(known.keys())}"
                )
            step_ranges = known[accum_type](str(request["date"]), request["step"])
            if len(step_ranges) == 0:
                raise ValueError(
                    f"No ranges matching type {accum_type} found in steps {request['step']}"
                )
            dim_config.pop("range_type")
            dim_config["coords"] = step_ranges
        else:
            dim_config.setdefault("coords", [[x] for x in request[dim]])


def increment_month(date: datetime, interval: int) -> Tuple[datetime, int]:
    replace = {"day": 1, "hour": 0}
    nmonth = date.month + 1
    if nmonth > 12:
        nmonth = nmonth % 12
        replace["year"] = date.year + 1
    replace["month"] = nmonth
    next_month = date.replace(**replace)
    delta = int((next_month - date).total_seconds() / (60 * 60 * interval))
    return next_month, delta


def get_interval(steps: List[int]) -> int:
    intervals = np.diff(np.array(steps))
    assert np.all(intervals == intervals[0]), "Step intervals must be constant"
    return int(intervals[0])


def monthly(date: str, steps: List[int], accumulated: bool = False) -> List[List[int]]:
    """
    Compute list of steps which belong to a specific month,
    starting from a given date and a list of forecast steps.
    The interval between forecast steps must be constant.
    """
    steps = sorted(list(set(steps)))
    if not all([isinstance(step, int) for step in steps]):
        raise ValueError("Steps for computing monthly ranges must be integers")
    interval = get_interval(steps)
    step_ranges = []
    step_index = 0 if steps[0] != 0 else 1
    fcdate = datetime.strptime(date, "%Y%m%d")
    start_month = fcdate + timedelta(hours=steps[step_index])
    if fcdate.day != 1:
        start_month, delta = increment_month(start_month, interval)
        step_index += delta
    while step_index < len(steps):
        next_month, delta = increment_month(start_month, interval)
        month_range = steps[step_index: step_index + delta + 1]

        # Only append range if we have the full months
        if len(month_range) == (delta + 1):
            step_ranges.append(
                {
                    "from": month_range[0],
                    "to": month_range[-1],
                    "by": delta if accumulated else interval,
                }
            )
        start_month = next_month
        step_index += delta
    return step_ranges


def weekly(date: str, steps: List[int]) -> List[List[int]]:
    """
    Compute list of steps which for weekly accumulations for the
    given list of forecast steps.
    """
    steps = sorted(list(set(steps)))
    if not all([isinstance(step, int) for step in steps]):
        raise ValueError("Steps for computing weekly ranges must be integers")

    print("STEPS", steps)
    interval = get_interval(steps)
    step_ranges = []
    start_day = (steps[0] // 24) * 24
    # Find first full day in steps
    if start_day not in steps:
        start_day += 24
    for start in range(start_day, steps[-1] - 168 + 1, 24):
        end = start + 168
        assert start in steps and end in steps
        step_ranges.append({"from": start, "to": end, "by": interval})
    return step_ranges
