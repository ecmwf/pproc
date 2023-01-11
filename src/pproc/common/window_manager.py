from typing import List
import numpy as np 

from common import Window, DiffWindow, SimpleOpWindow, WeightedSumWindow

def create_window(window_options, window_operation: str) -> Window:
    """
    Create window for specified window operations: min, max, sum, weightedsum and 
    diff. 

    :param start_step: start step of window interval
    :param end_step: end step of window interval
    :return: instance of the derived Window class for window operation
    :raises: ValueError for unsupported window operation string
    """
    if window_operation == 'diff':
        return DiffWindow(window_options)
    elif window_operation in ['min', 'max', 'sum']:
        return SimpleOpWindow(window_options, window_operation)
    elif window_operation == 'weightedsum':
        return WeightedSumWindow(window_options)
    raise ValueError(f'Unsupported window operation {window_operation}. Supported \
        types: diff, min, max, sum')

class WindowManager:
    """
    Class creating and managing active windows
    """
    def __init__(self, parameter):
        """
        Sort steps and create windows by reading in the config for specified parameter 

        :param parameter: config
        :raises: ValueError if creation of a window fails on deriving the window operation from
        the specified threshold comparison
        :raises: RuntimeError if no window operation was provided, or could be derived 
        """
        self.windows = []
        self.unique_steps = []
        for steps in parameter["steps"]:
            start_step = steps['start_step']
            end_step = steps['end_step']
            interval = steps['interval']
            write = steps.get('write', False)
            for step in range(start_step, end_step + 1, interval):
                if step not in self.unique_steps:
                    self.unique_steps.append(step)
                    if write:
                        self.windows.append(Window({'range': [step, step]}, 
                include_init=True))
        self.unique_steps = sorted(self.unique_steps)

        # Get window operation, or if not provided in config, derive from threshold
        if 'window_operation' in parameter:
            window_operation = parameter['window_operation']
        elif 'thresholds' in parameter:
            # Derive from threshold comparison parameter
            threshold_comparison = parameter['thresholds'][0]['comparison']
            if '<' in threshold_comparison:
                window_operation = 'min'
            elif '>' in threshold_comparison:
                window_operation = 'max'
            else:
                raise ValueError(f'No window_operation specified in config and unsupported derivation from \
                    threshold_comparison {threshold_comparison}')
        else:
            raise RuntimeError(f'No window operation specified, or could be derived')
        
        # Create windows from periods
        for period in parameter['periods']:
            self.windows.append(create_window(period, window_operation))

    def update_windows(self, step: int, data: np.array) -> List[Window]:
        """
        Updates all windows that include step with the step data values

        :param step: new step
        :param data: data for step
        :return: list of windows which have reached their end step
        """
        new_windows = []
        completed_windows = []
        for window in self.windows:
            window.add_step_values(step, data)

            if window.reached_end_step(step):
                completed_windows.append(window)
            else:
                new_windows.append(window)
        self.windows = new_windows
        return completed_windows

    def windows_completed(self) -> bool:
        """
        :return: boolean specifying if all windows have been completed 
        """
        return len(self.windows) == 0