import bisect
from typing import Iterator, List, Tuple
import numpy as np

from pproc.common import WindowManager, create_window
from pproc.common.accumulation import Accumulator, Coord


class ThresholdWindowManager(WindowManager):
    """
    Sort steps and create windows by reading in the config for specified parameter.
    Also, maintains dictionary of thresholds for each window.

    :param parameter: parameter config
    :raises: RuntimeError if no window operation was provided, or could be derived
    """

    def __init__(self, parameter, global_config):
        self.window_thresholds = {}
        WindowManager.__init__(self, parameter, global_config)

    @classmethod
    def window_operation_from_config(cls, window_config) -> str:
        """
        Derives window operation from config. If no window operation is explicitly
        specified then attempts to derive it from the thresholds - requires all
        comparison operators in the windows to be the same type.

        :param window_config: window configuration dictionary
        :return: string specifying window operation
        :raises: RuntimeError if no window operation could be derived
        """
        # Get window operation, or if not provided in config, derive from threshold
        window_operations = {}
        if "window_operation" in window_config:
            thresholds = window_config.get("thresholds", [])
            for threshold in thresholds:
                if isinstance(threshold["value"], str):
                    threshold["value"] = float(threshold["value"])
            window_operations[window_config["window_operation"]] = thresholds
        elif "thresholds" in window_config:
            # Derive from threshold comparison parameter
            for threshold in window_config["thresholds"]:
                if isinstance(threshold["value"], str):
                    threshold["value"] = float(threshold["value"])
                comparison = threshold["comparison"]
                if "<" in comparison:
                    operation = "minimum"
                elif ">" in comparison:
                    operation = "maximum"
                else:
                    raise RuntimeError(f"Unknown threshold comparison {comparison}")
                window_operations.setdefault(operation, []).append(threshold)

        if len(window_operations) == 0:
            raise RuntimeError(
                "Window with no operation specified, or none could be derived"
            )
        return window_operations

    def create_windows(self, parameter, global_config):

        windows = {}
        for window_index, window_config in enumerate(parameter["windows"]):
            window_operations = self.window_operation_from_config(window_config)

            for operation, thresholds in window_operations.items():
                for period in window_config["periods"]:
                    include_start = bool(window_config.get("include_start_step", False))
                    grib_keys = global_config.copy()
                    grib_keys.update(window_config.get("grib_set", {}))
                    window_acc, window_name = create_window(
                        period, operation, include_start, grib_keys, return_name=True
                    )
                    window_id = f"{window_name}_{operation}_{window_index}"
                    if window_id in windows:
                        raise Exception(f"Duplicate window {window_id}")
                    windows[window_id] = Accumulator({"step": window_acc})
                    self.window_thresholds[window_id] = thresholds
        return windows

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
    def __init__(self, parameter, global_config):
        ThresholdWindowManager.__init__(self, parameter, global_config)

    def create_windows(self, parameter, global_config):
        windows = super().create_windows(parameter, global_config)
        if "std_anomaly_windows" in parameter:
            # Create windows for standard anomaly
            for window_index, window_config in enumerate(
                parameter["std_anomaly_windows"]
            ):
                window_operations = self.window_operation_from_config(window_config)

                for operation, thresholds in window_operations.items():
                    for period in window_config["periods"]:
                        include_start = bool(
                            window_config.get("include_start_step", False)
                        )
                        grib_keys = global_config.copy()
                        grib_keys.update(window_config.get("grib_set", {}))
                        window_acc, window_name = create_window(
                            period,
                            operation,
                            include_start,
                            grib_keys,
                            return_name=True,
                        )
                        window_id = f"std_{window_name}_{operation}_{window_index}"
                        if window_id in windows:
                            raise Exception(f"Duplicate window {window_id}")
                        windows[window_id] = Accumulator({"step": window_acc})
                        self.window_thresholds[window_id] = thresholds
        return windows

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
