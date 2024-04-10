from dataclasses import dataclass
import numpy as np
from typing import Dict

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


class Window:
    """
    Class for collating data for all ensembles over an step interval
    """

    def __init__(self, window_config, accumulation: Accumulation):
        """
        :param window_config: window step configuration
        :param accumulation: accumulation done by the window
        """
        self.start = window_config.start
        self.end = window_config.end
        self.step = window_config.step
        self.name = window_config.name
        self.suffix = window_config.suffix
        self.steps = window_config.steps
        self.include_init = window_config.include_init

        self.acc = accumulation

        self.config_grib_header = {}

    def __contains__(self, step: AnyStep) -> bool:
        """
        :param step: current step
        :return: boolean specifying if step is in window interval
        """
        return step_to_coord(step) in self.acc

    def add_step_values(self, step: AnyStep, step_values: np.array):
        """
        Adds contribution of data values for specified step, if inside window, by computing
        reduction operation on existing step values and new step values - only the reduction
        operation on processed steps is stored

        :param step: step to update window with
        :param step_values: data values for step
        """
        self.acc.feed(step_to_coord(step), step_values)

    @property
    def step_values(self):
        values = self.acc.get_values()
        if values is None:
            return []
        return values

    def reached_end_step(self, step: AnyStep) -> bool:
        """
        :param step: current step
        :return: boolean specifying if current step is equal to window end step
        """
        return self.acc.is_complete()

    def size(self) -> int:
        """
        :return: size of window interval
        """
        return self.end - self.start

    def grib_header(self) -> Dict:
        """
        Returns window specific grib headers, including headers defined in
        config file

        :return: dictionary of header keys and values
        """
        header = {}
        if self.size() > 0 and self.end >= 256:
            # The range is encoded as two 8-bit integers
            header["unitOfTimeRange"] = 11

        header.update(self.config_grib_header)
        if self.size() == 0:
            header["step"] = self.name
        else:
            header.setdefault("stepType", "max")  # Don't override if set in config
            header["stepRange"] = self.name

        return header


def create_window(window_options, window_operation: str, include_start: bool) -> Window:
    """
    Create window for the given operation

    :param window_options: window range specification
    :param window operation: window operation: one of none, diff, add, minimum,
        maximum, weightedsum, diffdailyrate, mean, precomputed
    :return: Window instance that performs the operation
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

    acc = create_accumulation(
        {
            "operation": operation,
            "coords": coords,
            "sequential": True,
            **extra,
        }
    )
    return Window(config, acc)
