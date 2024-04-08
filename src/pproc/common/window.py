from dataclasses import dataclass
import numpy as np
from typing import Dict

from pproc.common.accumulation import (
    Aggregation,
    Difference,
    DifferenceRate,
    Mean,
    SimpleAccumulation,
    WeightedMean,
)
from pproc.common.steps import AnyStep, Step, step_to_coord


@dataclass
class WindowConfig:
    start: int
    end: int
    step: int
    name: str
    suffix: str
    steps: list


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
    return WindowConfig(start, end, step, name, suffix, steps)


class Window:
    """
    Class for collating data for all ensembles over an step interval
    """

    def __init__(self, window_options, include_init: bool = True):
        """
        :param window_options: specifies start and end step of window
        :param include_init: boolean specifying whether to include start step in window
        """
        config = parse_window_config(window_options, include_init)
        self.start = config.start
        self.end = config.end
        self.step = config.step
        self.name = config.name
        self.suffix = config.suffix
        self.steps = config.steps
        self.include_init = include_init

        self.config_grib_header = {}
        self.acc = Aggregation(
            [step_to_coord(step) for step in self.steps], sequential=True
        )

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


class SimpleOpWindow(Window):
    """
    Window with operation minimum, maximum, add
    """

    def __init__(
        self, window_options, window_operation: str, include_init: bool = False
    ):
        """
        :param window_options: config specifying start and end of window
        :param window_operation: name of reduction operation out of minimum, maximum, add
        :param include_init: boolean specifying whether to include start step
        """
        super().__init__(window_options, include_init)
        self.acc = SimpleAccumulation(
            window_operation,
            [step_to_coord(step) for step in self.steps],
            sequential=True,
        )


class WeightedSumWindow(SimpleOpWindow):
    """
    Window with weighted sum operation. Weighted sum is computed by weighting the
    data for each step by the step duration, and then dividing the sum by the total duration of the
    window.
    """

    def __init__(self, window_options):
        super().__init__(window_options, "add", include_init=False)
        self.acc = WeightedMean(self.start, self.steps)


class DiffWindow(Window):
    """
    Window with operation that takes difference between the end and start step. Only accepts data
    from these two steps
    """

    def __init__(self, window_options):
        super().__init__(window_options, include_init=True)
        self.steps = [self.start, self.end]
        self.acc = Difference(
            [step_to_coord(step) for step in self.steps], sequential=True
        )


class DiffDailyRateWindow(DiffWindow):
    """
    Window with operation that takes difference between end and start step and then divides difference
    by the total number of days in the window. Only accepts data for start and end step
    """

    def __init__(self, window_options):
        super().__init__(window_options)
        self.acc = DifferenceRate(self.steps, 1.0 / 24.0, sequential=True)


class MeanWindow(SimpleOpWindow):
    """
    Window with operation that computes mean over the steps in window
    """

    def __init__(self, window_options, include_init=False):
        super().__init__(window_options, "add", include_init=include_init)
        self.acc = Mean([step_to_coord(step) for step in self.steps], sequential=True)


class PrecomputedWindow(Window):
    """
    Window containing a single pre-computed accumulation with the given step range
    """

    def __init__(self, window_options):
        super().__init__(window_options, include_init=True)
        self.steps = [Step(self.start, self.end)]
        self.acc = Aggregation(
            [step_to_coord(step) for step in self.steps], sequential=True
        )
