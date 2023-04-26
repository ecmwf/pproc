import bisect
from typing import Iterator, List
import numpy as np

from pproc.common import WindowManager, Window, create_window


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
                    operation = "min"
                elif ">" in comparison:
                    operation = "max"
                else:
                    raise RuntimeError(f"Unknown threshold comparison {comparison}")
                window_operations.setdefault(operation, []).append(threshold)

        if len(window_operations) == 0:
            raise RuntimeError(
                "Window with no operation specified, or none could be derived"
            )
        return window_operations

    def create_windows(self, parameter, global_config):

        for window_index, window_config in enumerate(parameter["windows"]):
            window_operations = self.window_operation_from_config(window_config)

            for operation, thresholds in window_operations.items():
                for period in window_config["periods"]:
                    include_start = bool(window_config.get("include_start_step", False))
                    new_window = create_window(period, operation, include_start)
                    new_window.config_grib_header = global_config.copy()
                    new_window.config_grib_header.update(
                        window_config.get("grib_set", {})
                    )
                    window_id = f"{new_window.name}_{operation}_{window_index}"
                    if window_id in self.windows:
                        raise Exception(f"Duplicate window {window_id}")
                    self.windows[window_id] = new_window
                    self.window_thresholds[window_id] = thresholds

    def thresholds(self, identifier):
        """
        Returns thresholds for window and deletes window from window:threshold dictionary
        """
        return self.window_thresholds.pop(identifier)

    def update_from_checkpoint(self, checkpoint_step: int):
        """
        Find the earliest start step for windows not completed by
        checkpoint and update list of unique steps. Remove all
        completed windows and their associated thresholds.

        :param checkpoint_step: step reached at last checkpoint
        """
        deleted_windows = super().update_from_checkpoint(checkpoint_step)
        for window in deleted_windows:
            del self.window_thresholds[window]

    def delete_windows(self, window_ids: List[str]):
        super().delete_windows(window_ids)
        for window_id in window_ids:
            del self.thresholds[window_id]


class AnomalyWindowManager(ThresholdWindowManager):
    def __init__(self, parameter, global_config):
        ThresholdWindowManager.__init__(self, parameter, global_config)

    def create_windows(self, parameter, global_config):
        super().create_windows(parameter, global_config)
        if "std_anomaly_windows" in parameter:
            # Create windows for standard anomaly
            for window_index, window_config in enumerate(parameter["std_anomaly_windows"]):
                window_operations = self.window_operation_from_config(window_config)

                for operation, thresholds in window_operations.items():
                    for period in window_config["periods"]:
                        include_start = bool(
                            window_config.get("include_start_step", False)
                        )
                        new_window = create_window(period, operation, include_start)
                        new_window.config_grib_header = global_config.copy()
                        new_window.config_grib_header.update(
                            window_config.get("grib_set", {})
                        )
                        window_id = f"std_{new_window.name}_{operation}_{window_index}"
                        if window_id in self.windows:
                            raise Exception(f"Duplicate window {window_id}")
                        self.windows[window_id] = new_window
                        self.window_thresholds[window_id] = thresholds

    def update_windows(
        self, step, data: np.array, clim_mean: np.array, clim_std: np.array
    ) -> Iterator[Window]:
        """
        Updates all windows that include step with either the anomaly with clim_mean
        or standardised anomaly including clim_std. Function modifies input data array.

        :param step: new step
        :param data: data for step
        :param clim_mean: mean from climatology
        :param clim_std: standard deviation from climatology
        :return: generator for completed windows
        """
        anomaly = data - clim_mean
        std_anomaly = anomaly / clim_std
        for identifier, window in list(self.windows.items()):
            if identifier.split('_')[0] == "std":
                window.add_step_values(step, std_anomaly)
            else:
                window.add_step_values(step, anomaly)

            if window.reached_end_step(step):
                yield identifier, self.windows.pop(identifier)
