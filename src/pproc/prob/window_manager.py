from pproc.common import WindowManager, create_window

class ThresholdWindowManager(WindowManager):
    """
    Sort steps and create windows by reading in the config for specified parameter. 
    Also, maintains dictionary of thresholds for each window.

    :param parameter: parameter config
    :raises: RuntimeError if no window operation was provided, or could be derived
    """
    def __init__(self, parameter):
        self.window_thresholds = {}
        WindowManager.__init__(self, parameter)

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
            window_operations[window_config["window_operation"]] = window_config["thresholds"]
        elif "thresholds" in window_config:
            # Derive from threshold comparison parameter
            for threshold in window_config["thresholds"]:
                comparison = threshold["comparison"]
                if "<" in comparison:
                    operation = "min"
                elif ">" in comparison:
                    operation = "max"
                else:
                    raise RuntimeError(
                        f"Unknown threshold comparison {comparison}"
                    )
                window_operations.setdefault(operation, []).append(threshold)

        if len(window_operations) == 0:
            raise RuntimeError(
                f"Window  with no operation specified, or none could be derived"
            )
        return window_operations

    def create_windows(self, parameter):
        for window_config in parameter["windows"]:
            window_operations = self.window_operation_from_config(window_config)

            for operation, thresholds in window_operations.items():
                for period in window_config["periods"]:
                    new_window = create_window(period, operation)
                    new_window.config_grib_header = window_config.get("grib_set", {})
                    self.windows.append(new_window)
                    self.window_thresholds[new_window] = thresholds

    def thresholds(self, window):
        """
        Returns thresholds for window and deletes window from window:threshold dictionary
        """
        return self.window_thresholds.pop(window)