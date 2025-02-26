from typing import Iterator, List, Tuple, Optional
import numpy as np

from pproc.common.window_manager import WindowManager
from pproc.common.accumulation import Accumulator, Coord
from pproc.common.accumulation_manager import AccumulationManager
from pproc.config.accumulation import LegacyStepAccumulation


class ThresholdWindowManager(WindowManager):
    """
    Sort steps and create windows by reading in the config for specified parameter.
    Also, maintains dictionary of thresholds for each window.

    :param parameter: parameter config
    :raises: RuntimeError if no window operation was provided, or could be derived
    """

    def __init__(
        self,
        accumulations: dict[str, LegacyStepAccumulation],
        metadata: Optional[dict] = None,
    ):
        self.window_thresholds = {}
        for window_id, config in accumulations["step"].make_configs(metadata):
            thresholds = config.pop("thresholds", [])
            if not thresholds:
                raise RuntimeError(
                    "Window with no operation specified, or none could be derived"
                )
            self.window_thresholds[window_id] = thresholds
        self.mgr = AccumulationManager.create(accumulations, metadata)

    def thresholds(self, identifier):
        """
        Returns thresholds for window and deletes window from window:threshold dictionary
        """
        return self.window_thresholds.pop(identifier)

    def delete_windows(self, window_ids: List[str]) -> Coord:
        new_start = super().delete_windows(window_ids)
        for window_id in window_ids:
            del self.window_thresholds[window_id]
        return new_start


class AnomalyWindowManager(ThresholdWindowManager):
    def update_windows(
        self, keys: dict, data: np.array, clim_mean: np.array, clim_std: np.array
    ) -> Iterator[Tuple[str, Accumulator]]:
        """
        Updates all windows that include the given keys with either the anomaly
        with clim_mean or standardised anomaly including clim_std. Function
        modifies input data array.

        :param keys: keys identifying the new chunk of data
        :param data: data chunk
        :param clim_mean: mean from climatology
        :param clim_std: standard deviation from climatology
        :return: generator for completed windows
        """
        anomaly = data - clim_mean
        std_anomaly = anomaly / clim_std
        for identifier, accum in list(self.mgr.accumulations.items()):
            if identifier.split("_")[0] == "std":
                processed = accum.feed(keys, std_anomaly)
            else:
                processed = accum.feed(keys, anomaly)

            if processed and accum.is_complete():
                yield identifier, self.mgr.accumulations.pop(identifier)
