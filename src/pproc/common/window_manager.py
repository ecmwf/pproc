from typing import Iterator

import numpy as np

from pproc.common import (
    Window,
    SimpleOpWindow,
    WeightedSumWindow,
    DiffWindow,
    DiffDailyRateWindow,
)


def create_window(window_options, window_operation: str) -> Window:
    """
    Create window for specified window operations: min, max, sum, weightedsum and
    diff.

    :param start_step: start step of window interval
    :param end_step: end step of window interval
    :return: instance of the derived Window class for window operation
    :raises: ValueError for unsupported window operation string
    """
    if window_options["range"][0] == window_options["range"][1]:
        return Window(window_options, include_init=True)
    if window_operation == "diff":
        return DiffWindow(window_options)
    if window_operation in ["min", "max", "sum"]:
        return SimpleOpWindow(window_options, window_operation, include_init=False)
    if window_operation == "weightedsum":
        return WeightedSumWindow(window_options)
    if window_operation == "diffdailyrate":
        return DiffDailyRateWindow(window_options)
    raise ValueError(
        f"Unsupported window operation {window_operation}. "
        + "Supported types: diff, min, max, sum, weightedsum, diffdailyrate"
    )


class WindowManager:
    """
    Class for creating and managing active windows
    """

    def __init__(self, parameter):
        """
        Sort steps and create windows by reading in the config for specified parameter

        :param parameter: parameter config
        :raises: RuntimeError if no window operation was provided, or could be derived
        """
        self.windows = []
        self.unique_steps = set()
        for steps in parameter["steps"]:
            start_step = steps["start_step"]
            end_step = steps["end_step"]
            interval = steps["interval"]

            for step in range(start_step, end_step + 1, interval):
                if step not in self.unique_steps:
                    self.unique_steps.add(step)

        self.unique_steps = sorted(self.unique_steps)
        self.create_windows(parameter)

    def create_windows(self, parameter):
        """
        Creates windows from parameter config and specified window operation
        """
        for window_config in parameter["windows"]:
            for period in window_config["periods"]:
                new_window = create_window(period, window_config["window_operation"])
                new_window.config_grib_header = window_config.get("grib_set", {})
                self.windows.append(new_window)

    def update_windows(self, step: int, data: np.array) -> Iterator[Window]:
        """
        Updates all windows that include step with the step data values

        :param step: new step
        :param data: data for step
        :return: generator for completed windows
        """
        new_windows = []
        for window in self.windows:
            window.add_step_values(step, data)

            if window.reached_end_step(step):
                yield window
            else:
                new_windows.append(window)
        self.windows = new_windows
