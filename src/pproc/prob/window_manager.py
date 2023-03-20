import bisect
from typing import Iterator
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
        for window_config in parameter["windows"]:
            window_operations = self.window_operation_from_config(window_config)

            for operation, thresholds in window_operations.items():
                for period in window_config["periods"]:
                    include_start = bool(window_config.get("include_start_step", False))
                    new_window = create_window(period, operation, include_start)
                    new_window.config_grib_header = global_config.copy()
                    new_window.config_grib_header.update(
                        window_config.get("grib_set", {})
                    )
                    self.windows.append(new_window)
                    self.window_thresholds[new_window] = thresholds

    def thresholds(self, window):
        """
        Returns thresholds for window and deletes window from window:threshold dictionary
        """
        return self.window_thresholds.pop(window)

    def update_from_checkpoint(self, checkpoint_step: int):
        """
        Find the earliest start step for windows containing checkpoint_step

        """
        new_start_step = checkpoint_step + 1
        delete_windows = []
        for window in self.windows:
            real_start = window.start + int(window.include_init)
            if checkpoint_step >= window.end:
                delete_windows.append(window)
            elif checkpoint_step in window and real_start < new_start_step:
                new_start_step = real_start

        for window in delete_windows:
            self.window_thresholds.pop(window)
            self.windows.remove(window)
        start_index = bisect.bisect_left(self.unique_steps, new_start_step)
        self.unique_steps = self.unique_steps[start_index:]


class AnomalyWindowManager(ThresholdWindowManager):
    def __init__(self, parameter, global_config):
        self.standardised_anomaly_windows = []
        ThresholdWindowManager.__init__(self, parameter, global_config)

    def create_windows(self, parameter, global_config):
        super().create_windows(parameter, global_config)
        if "std_anomaly_windows" in parameter:
            # Create windows for standard anomaly
            for window_config in parameter["std_anomaly_windows"]:
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
                        self.standardised_anomaly_windows.append(new_window)
                        self.window_thresholds[new_window] = thresholds

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
        data = data - clim_mean
        new_anom_windows = []
        for window in self.windows:
            window.add_step_values(step, data)

            if window.reached_end_step(step):
                yield window
            else:
                new_anom_windows.append(window)
        self.windows = new_anom_windows

        new_std_anom_windows = []
        data = data / clim_std
        for window in self.standardised_anomaly_windows:
            window.add_step_values(step, data)

            if window.reached_end_step(step):
                yield window
            else:
                new_std_anom_windows.append(window)
        self.standardised_anomaly_windows = new_std_anom_windows

    def update_from_checkpoint(self, checkpoint_step: int):
        """
        Find the earliest start step for windows containing checkpoint_step

        """
        new_start_step = checkpoint_step + 1
        for window_set in [self.windows, self.standardised_anomaly_windows]:
            delete_windows = []
            for window in window_set:
                real_start = window.start + int(window.include_init)
                if checkpoint_step in window and real_start < new_start_step:
                    new_start_step = real_start
                if checkpoint_step >= window.end:
                    delete_windows.append(window)

            for window in delete_windows:
                self.window_thresholds.pop(window)
                window_set.remove(window)
        start_index = bisect.bisect_left(self.unique_steps, new_start_step)
        self.unique_steps = self.unique_steps[start_index:]
