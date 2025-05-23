# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Iterator, List, Tuple, Optional, Dict
import numpy as np

from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.config.accumulation import accumulation_factory


class ThresholdAccumulationManager(AccumulationManager):
    """
    Maintains dictionary of thresholds for each window.

    :param parameter: parameter config
    :raises: RuntimeError if no window operation was provided, or could be derived
    """

    _thresholds: dict

    @classmethod
    def create(cls, config: Dict[str, dict], grib_keys: Optional[dict] = None):
        all_thresholds = {}
        step_configs = config["step"]
        if isinstance(step_configs, dict):
            step_configs = accumulation_factory(step_configs)
        for window_id, step_cfg in step_configs.make_configs(grib_keys):
            thresholds = step_cfg.pop("thresholds", [])
            if not thresholds:
                raise ValueError("Step accumulation does not contain thresholds")
            all_thresholds[window_id] = thresholds

        mgr = super().create(config, grib_keys)
        mgr._thresholds = all_thresholds
        return mgr

    def thresholds(self, identifier: str) -> dict:
        """
        Returns thresholds for window and deletes window from window:threshold dictionary
        """
        return self._thresholds.pop(identifier)

    def delete(self, accumulations: List[str]):
        super().delete(accumulations)
        for accum_id in accumulations:
            del self._thresholds[accum_id]


class AnomalyAccumulationManager(ThresholdAccumulationManager):
    def feed(
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
        for identifier, accum in list(self.accumulations.items()):
            if identifier.split("_")[0] == "std":
                processed = accum.feed(keys, std_anomaly)
            else:
                processed = accum.feed(keys, anomaly)

            if processed and accum.is_complete():
                yield identifier, self.accumulations.pop(identifier)
