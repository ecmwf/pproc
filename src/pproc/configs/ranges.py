from datetime import datetime
from typing import List

import numpy as np


def populate_accums(accums: dict, request: dict) -> dict:
    for dim, dim_config in accums.items():
        if dim == "step":
            accum_type = dim_config.get("type", None)
            if accum_type == "monthly":
                step_ranges = monthly(
                    str(request["date"]), list(map(int, request["step"]))
                )
                if len(step_ranges) == 0:
                    raise ValueError(f"No full months found in steps {request['step']}")
                dim_config.pop("type")
                dim_config["coords"] = step_ranges
            elif accum_type == "weekly":
                step_ranges = weekly(list(map(int, request["step"])))
                if len(step_ranges) == 0:
                    raise ValueError(f"No full months found in steps {request['step']}")
                dim_config.pop("type")
                dim_config["coords"] = step_ranges


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
    steps = sorted(list(set(steps)))
    if all([isinstance(step, int) for step in steps]):
        intervals = np.diff(np.array(steps))
        assert np.all(intervals == intervals[0]), "Step intervals must be constant"
        interval = intervals[0]
        start_month = datetime.strptime(date, "%Y%m%d")
        step_ranges = []
        step_index = 0 if steps[0] != 0 else 1
        while step_index < len(steps):
            next_month = _increment_month(start_month)
            delta = int(
                (next_month - start_month).total_seconds() / (60 * 60 * interval)
            )
            month_range = steps[step_index: step_index + delta]
            if start_month.day == 1 and len(month_range) == delta:
                # Only append range if we have the full months
                step_ranges.append(month_range)
            start_month = next_month
            step_index += delta
        return step_ranges


def weekly(steps: List[int]) -> List[List[int]]:
    """
    Compute list of steps which for weekly accumulations for the
    given list of forecast steps.
    """
    steps = sorted(list(set(steps)))
    if all([isinstance(step, int) for step in steps]):
        step_ranges = []
        for start in range(steps[0], steps[-1] - 168 + 1, 24):
            end = start + 168
            step_ranges.append(steps[steps.index(start): steps.index(end) + 1])
        return step_ranges
    if not all([isinstance(step, str) for step in steps]):
        raise ValueError("Steps must be the same type, either integers or strings")
    step_ranges = []
    for step_range in steps:
        start, end = map(int, step_range.split("-"))
        if (end - start) != 168:
            raise ValueError("Weekly ranges must be 168 hours long")
        step_ranges.append([step_range])
    return step_ranges
