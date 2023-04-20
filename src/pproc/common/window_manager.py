from typing import Iterator, List
import bisect

import numpy as np

from pproc.common import (
    Window,
    SimpleOpWindow,
    WeightedSumWindow,
    DiffWindow,
    DiffDailyRateWindow,
    MeanWindow,
)


def create_window(window_options, window_operation: str, include_start: bool) -> Window:
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
    if window_operation in ["min", "max", "sum", "concatenate"]:
        return SimpleOpWindow(window_options, window_operation, include_start)
    if window_operation == "weightedsum":
        return WeightedSumWindow(window_options)
    if window_operation == "diffdailyrate":
        return DiffDailyRateWindow(window_options)
    if window_operation == "mean":
        return MeanWindow(window_options, include_start)
    raise ValueError(
        f"Unsupported window operation {window_operation}. "
        + "Supported types: diff, min, max, sum, weightedsum, diffdailyrate"
    )


class WindowManager:
    """
    Class for creating and managing active windows
    """

    def __init__(self, parameter, global_config):
        """
        Sort steps and create windows by reading in the config for specified parameter

        :param parameter: parameter config
        :param global_config: global dictionary of key values for grib_set in all windows
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
        self.create_windows(parameter, global_config)

    def create_windows(self, parameter, global_config):
        """
        Creates windows from parameter config and specified window operation
        """
        for window_config in parameter["windows"]:
            for period in window_config["periods"]:
                include_start = bool(window_config.get("include_start_step", False))
                new_window = create_window(
                    period, window_config["window_operation"], include_start
                )
                new_window.config_grib_header = global_config.copy()
                new_window.config_grib_header.update(window_config.get("grib_set", {}))
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

    def update_from_checkpoint(self, checkpoint_step: int):
        """
        Find the earliest start step for windows not completed by
        checkpoint and update list of unique steps. Remove all
        completed windows and their associated thresholds.

        :param checkpoint_step: step reached at last checkpoint
        """
        new_start_step = checkpoint_step + 1
        delete_windows = []
        for window in self.windows:
            real_start = window.start + int(not window.include_init)
            if checkpoint_step >= window.end:
                delete_windows.append(window)
            elif real_start < new_start_step:
                new_start_step = real_start

        for window in delete_windows:
            self.windows.remove(window)
        start_index = bisect.bisect_left(self.unique_steps, new_start_step)
        self.unique_steps = self.unique_steps[start_index:]

    def delete_windows(self, window_names: List[str]):
        """
        Remove windows in the list of provided window names and updates steps
        to only those contained in remaining list of windows
        
        :param window_names: list of window names to delete
        """
        for window in self.windows:
            if window.name in window_names:
                self.windows.remove(window)

        for step in self.unique_steps:
            in_any_window = np.any([step in window for window in self.windows])
            if in_any_window:
                # Steps must be processed in order so stop at first step that appears
                # in remaining window
                break
            self.unique_steps.remove(step)
