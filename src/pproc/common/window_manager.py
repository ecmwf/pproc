from typing import Iterator, List, Tuple
import bisect

import numpy as np

from pproc.common.accumulation import Accumulation
from pproc.common.steps import AnyStep, Step, parse_step, step_to_coord
from pproc.common.window import create_window


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
        self.create_windows(parameter, global_config)
        if "steps" not in parameter:
            for accum in self.windows.values():
                self.unique_steps.update(accum.coords)
        else:
            for steps in parameter["steps"]:
                start_step = steps["start_step"]
                end_step = steps["end_step"]
                interval = steps["interval"]
                range_len = steps.get("range", None)

                if range_len is None:
                    self.unique_steps.update(range(start_step, end_step + 1, interval))
                else:
                    for sstep in range(start_step, end_step - range_len + 1, interval):
                        self.unique_steps.add(
                            step_to_coord(Step(sstep, sstep + range_len))
                        )

        self.unique_steps = sorted(self.unique_steps, key=parse_step)

    def create_windows(self, parameter, global_config):
        """
        Creates windows from parameter config and specified window operation
        """
        for window_index, window_config in enumerate(parameter["windows"]):
            for period in window_config["periods"]:
                include_start = bool(window_config.get("include_start_step", False))
                grib_keys = global_config.copy()
                grib_keys.update(window_config.get("grib_set", {}))
                window_acc, window_name = create_window(
                    period,
                    window_config.get("window_operation", "none"),
                    include_start,
                    grib_keys,
                    return_name=True,
                )
                window_id = f"{window_name}_{window_index}"
                if window_id in self.windows:
                    raise Exception(f"Duplicate window {window_id}")
                self.windows[window_id] = window_acc

    def update_windows(
        self, step: AnyStep, data: np.array
    ) -> Iterator[Tuple[str, Accumulation]]:
        """
        Updates all windows that include step with the step data values

        :param step: new step
        :param data: data for step
        :return: generator for completed windows
        """
        for identifier, accum in list(self.windows.items()):
            accum.feed(step_to_coord(step), data)

            if accum.is_complete():
                completed = self.windows.pop(identifier)
                yield identifier, completed
                del completed

    def delete_windows(self, window_ids: List[str]):
        """
        Remove windows in the list of provided window identifiers and updates steps
        to only those contained in remaining list of windows

        :param window_ids: list of identifiers of windows to delete
        """
        for identifier in window_ids:
            del self.windows[identifier]

        for step_index, step in enumerate(self.unique_steps):
            in_any_window = any(step in accum for accum in self.windows.values())
            if in_any_window:
                # Steps must be processed in order so stop at first step that appears
                # in remaining window
                break
        self.unique_steps[:] = self.unique_steps[step_index:]
