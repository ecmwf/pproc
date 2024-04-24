from dataclasses import dataclass
import numpy as np
from typing import Dict, Optional, Tuple, Union

from pproc.common.accumulation import Accumulation, create_accumulation
from pproc.common.steps import AnyStep, Step, step_to_coord


@dataclass
class WindowConfig:
    start: int
    end: int
    step: int
    name: str
    suffix: str
    steps: list
    include_init: bool


def parse_window_config(config: dict, include_init: bool = True) -> WindowConfig:
    start = int(config["range"][0])
    end = int(config["range"][1])
    step = int(config["range"][2]) if len(config["range"]) > 2 else 1
    window_size = end - start
    name = str(end) if window_size == 0 else f"{start}-{end}"
    suffix = f"{window_size:0>3}_{start:0>3}h_{end:0>3}h"
    if include_init:
        steps = list(range(start, end + 1, step))
    else:
        steps = list(range(start + step, end + 1, step))
    return WindowConfig(start, end, step, name, suffix, steps, include_init)


def create_window(
    window_options,
    window_operation: str,
    include_start: bool,
    grib_keys: Optional[dict] = None,
    return_name: bool = False,
) -> Union[Accumulation, Tuple[Accumulation, str]]:
    """
    Create window for the given operation

    :param window_options: window range specification
    :param window operation: window operation: one of none, diff, add, minimum,
        maximum, weightedsum, diffdailyrate, mean, precomputed
    :param grib_keys: additional grib keys to tie to the window
    :param return_name: if True, return the window name as well
    :return: Window instance that performs the operation, window name (only if
        `return_name` is True)
    :raises: ValueError for unsupported window operation string
    """
    include_init = (
        window_options["range"][0] == window_options["range"][1]
    ) or include_start

    config = None
    operation = None
    coords = None
    extra = {}
    if window_operation == "none":
        config = parse_window_config(window_options, include_init)
        if len(config.steps) > 1:
            raise ValueError(
                "Window operation can not be none for windows containing more than a single step"
            )
        operation = "aggregation"
    elif window_operation == "diff":
        config = parse_window_config(window_options, True)
        config.steps = [config.start, config.end]
        operation = "difference"
    elif window_operation == "add":
        config = parse_window_config(window_options, include_init)
        operation = "sum"
    elif window_operation in ["minimum", "maximum"]:
        config = parse_window_config(window_options, include_init)
        operation = window_operation
    elif window_operation == "weightedsum":
        config = parse_window_config(window_options, False)
        operation = "weighted_mean"
        coords = [config.start] + config.steps
    elif window_operation == "diffdailyrate":
        config = parse_window_config(window_options, True)
        config.steps = [config.start, config.end]
        extra["factor"] = 1.0 / 24.0
        operation = "difference_rate"
    elif window_operation == "mean":
        config = parse_window_config(window_options, include_init)
        operation = "mean"
    elif window_operation == "precomputed":
        config = parse_window_config(window_options, True)
        config.steps = [Step(config.start, config.end)]
        operation = "aggregation"

    if config is None:
        raise ValueError(
            f"Unsupported window operation {window_operation}. Supported types: "
            + "diff, minimum, maximum, add, weightedsum, diffdailyrate, mean and precomputed"
        )

    if coords is None:
        coords = [step_to_coord(step) for step in config.steps]

    grib_header = {}
    if config.end > config.start and config.end >= 256:
        # The range is encoded as two 8-bit integers
        grib_header["unitOfTimeRange"] = 11

    if config.end == config.start:
        if config.end >= 256:
            grib_header["timeRangeIndicator"] = 10
        elif config.end == 0:
            grib_header["timeRangeIndicator"] = 1
        else:
            grib_header["timeRangeIndicator"] = 0

    if grib_keys is not None:
        grib_header.update(grib_keys)

    if config.end == config.start:
        grib_header["step"] = config.name
    else:
        grib_header.setdefault("stepType", "max")  # Don't override if set in config
        grib_header["stepRange"] = config.name

    acc = create_accumulation(
        {
            "operation": operation,
            "coords": coords,
            "sequential": True,
            "grib_keys": grib_header,
            **extra,
        }
    )

    if return_name:
        return acc, config.name
    return acc
