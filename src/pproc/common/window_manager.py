from typing import Dict, Iterator, List, Tuple
import bisect

import numpy as np

from pproc.common.accumulation import Accumulator, Coord
from pproc.common.accumulation_manager import AccumulationManager
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
        unique_steps = set()
        windows = self.create_windows(parameter, global_config)
        if "steps" not in parameter:
            for accum in windows.values():
                unique_steps.update(accum["step"].coords)
        else:
            for steps in parameter["steps"]:
                start_step = steps["start_step"]
                end_step = steps["end_step"]
                interval = steps["interval"]
                range_len = steps.get("range", None)

                if range_len is None:
                    unique_steps.update(range(start_step, end_step + 1, interval))
                else:
                    for sstep in range(start_step, end_step - range_len + 1, interval):
                        unique_steps.add(
                            step_to_coord(Step(sstep, sstep + range_len))
                        )

        self.mgr = AccumulationManager({})
        self.mgr.accumulations = windows
        self.mgr.coords = {"step": unique_steps}

    def create_windows(self, parameter, global_config):
        """
        Creates windows from parameter config and specified window operation
        """
        windows = {}
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
                if window_id in windows:
                    raise Exception(f"Duplicate window {window_id}")
                windows[window_id] = Accumulator({"step": window_acc})
        return windows

    @property
    def dims(self) -> Dict[str, List[Coord]]:
        sorted_dims = {}
        for key, coords in self.mgr.coords.items():
            if key == "step":
                sorted_dims[key] = sorted(coords, key=parse_step)
            else:
                sorted_dims[key] = sorted(coords)
        return sorted_dims

    def update_windows(
        self, keys: Dict[str, Coord], data: np.ndarray
    ) -> Iterator[Tuple[str, Accumulator]]:
        """
        Updates all windows containing the given keys with the associated data

        :param keys: keys identifying the new chunk of data
        :param data: data chunk
        :return: generator for completed windows
        """
        yield from self.mgr.feed(keys, data)

    def delete_windows(self, window_ids: List[str]) -> Coord:
        """
        Remove windows in the list of provided window identifiers and updates steps
        to only those contained in remaining list of windows

        :param window_ids: list of identifiers of windows to delete
        :return: new first step
        """
        self.mgr.delete(window_ids)
        return min(self.mgr.coords["step"], key=parse_step)
