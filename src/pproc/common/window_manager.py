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
        self.windows = {}
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
        for window_index, window_config in enumerate(parameter["windows"]):
            for period in window_config["periods"]:
                include_start = bool(window_config.get("include_start_step", False))
                new_window = create_window(
                    period, window_config["window_operation"], include_start
                )
                new_window.config_grib_header = global_config.copy()
                new_window.config_grib_header.update(window_config.get("grib_set", {}))
                window_id = f"{new_window.name}_{window_index}"
                if window_id in self.windows:
                    raise Exception(f"Duplicate window {window_id}")
                self.windows[window_id] = new_window

    def update_windows(self, step: int, data: np.array) -> Iterator[Window]:
        """
        Updates all windows that include step with the step data values

        :param step: new step
        :param data: data for step
        :return: generator for completed windows
        """
        for identifier, window in list(self.windows.items()):
            window.add_step_values(step, data)

            if window.reached_end_step(step):
                yield identifier, self.windows.pop(identifier)

    def update_from_checkpoint(self, checkpoint_step: int) -> List[str]:
        """
        Find the earliest start step for windows not completed by
        checkpoint and update list of unique steps. Remove all
        completed windows and their associated thresholds.

        :param checkpoint_step: step reached at last checkpoint
        :return: list of deleted window identifiers
        """
        new_start_step = checkpoint_step + 1
        deleted_windows = []
        for identifier, window in list(self.windows.items()):
            real_start = window.start + int(not window.include_init)
            if checkpoint_step >= window.end:
                del self.windows[identifier]
                deleted_windows.append(identifier)
            elif real_start < new_start_step:
                new_start_step = real_start

        start_index = bisect.bisect_left(self.unique_steps, new_start_step)
        self.unique_steps = self.unique_steps[start_index:]
        return deleted_windows

    def delete_windows(self, window_ids: List[str]):
        """
        Remove windows in the list of provided window identifiers and updates steps
        to only those contained in remaining list of windows
        
        :param window_ids: list of identifiers of windows to delete
        """
        for identifier in window_ids:
            del self.windows[identifier]

        for step in self.unique_steps:
            in_any_window = np.any([step in window for window in self.windows.values()])
            if in_any_window:
                # Steps must be processed in order so stop at first step that appears
                # in remaining window
                break
            self.unique_steps.remove(step)
