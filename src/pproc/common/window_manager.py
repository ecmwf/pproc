from typing import Dict, Iterator, List, Tuple

import numpy as np
from typing import Optional

from pproc.common.accumulation import Accumulator, Coord
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.steps import parse_step
from pproc.config.accumulation import LegacyStepAccumulation


class WindowManager:
    """
    Class for creating and managing active windows
    """

    def __init__(self, accumulations: dict, metadata: Optional[dict] = None):
        """
        Sort steps and create windows by reading in the config for specified parameter

        :accumulations dict: accumulation config
        :metadata: dictionary of key values for grib_set in all windows
        :raises: RuntimeError if no window operation was provided, or could be derived
        """
        if "windows" in accumulations:
            accumulations = {"step": LegacyStepAccumulation(**accumulations)}
        self.mgr = AccumulationManager.create(accumulations, metadata)

    @property
    def dims(self) -> Dict[str, List[Coord]]:
        return self.mgr.sorted_coords({"step": parse_step})

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
        steps = self.mgr.coords["step"]
        if len(steps) == 0:
            return None
        return min(steps, key=parse_step)
