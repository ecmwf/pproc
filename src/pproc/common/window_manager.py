from typing import List
import numpy as np 

from common import Window, DiffWindow, SimpleOpWindow, WeightedSumWindow

def create_window(start_step: int, end_step: int, window_operation) -> Window:
    window_options = {'range': [start_step, end_step]}
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
    Class creating and manage lifespan of windows
    """
    def __init__(self, parameter):
        # Sort steps and create instantaneous windows by reading in the config 
        # for specified parameter
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

        # Create windows from periods
        for periods in parameter['periods']:
            if 'window_operation' in parameter:
                window_operation = parameter['window_operation']
            else:
                # Derive from threshold comparison parameter
                threshold_comparison = parameter['threshold_comparison']
                if '<' in threshold_comparison:
                    window_operation = 'min'
                elif '>' in threshold_comparison:
                    window_operation = 'max'
                else:
                    raise ValueError(f'No window_operation specified in config and unsupported derivation from \
                        threshold_comparison {threshold_comparison}')
            self.windows.append(create_window(periods['start_step'], periods['end_step'], window_operation))

        self.unique_steps = sorted(self.unique_steps)

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
        Returns if all windows have been completed 
        """
        return len(self.windows) == 0