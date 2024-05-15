from typing import Dict, Iterator, List, Tuple

import numpy as np

from pproc.common.accumulation import Accumulator, Coord
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.steps import parse_step
from pproc.common.window import legacy_window_factory


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
        for accum in windows.values():
            unique_steps.update(accum["step"].coords)

        self.mgr = AccumulationManager({})
        self.mgr.accumulations = windows
        self.mgr.coords = {"step": unique_steps}

    def create_windows(self, parameter, global_config) -> Dict[str, Accumulator]:
        """
        Creates windows from parameter config and specified window operation
        """
        windows = {}
        for window_id, acc_config in legacy_window_factory(parameter, global_config):
            if window_id in windows:
                raise Exception(f"Duplicate window {window_id}")
            windows[window_id] = Accumulator.create({"step": acc_config})
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
