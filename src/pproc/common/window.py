import numexpr
import numpy as np

class Window:
    """
    Class for collating data for all ensembles over an step interval for computing
    time-averaged probabilities
    """
    def __init__(self, window_options, include_init: bool = True):

        """
        :param window_options: specifies start and end step of window
        :param include_init: boolean specifying whether to include start step in window
        """
        self.start = int(window_options['range'][0])
        self.end = int(window_options['range'][1])
        self.include_init = include_init
        window_size = self.end-self.start
        self.suffix = f"{window_size:0>3}_{self.start:0>3}h_{self.end:0>3}h"
        self.name = f"{self.start}-{self.end}"
        self.operation = None
        self.step_values = []

    def set_reduction_operation(self, operation):
        """
        Sets reduction operation on existing step data values and new step data values. 
        Accepts string arguments min, max, sum or a lambda function of the form
        f(current_step_values, new_step_values) -> reduced_step_values
        """
        if isinstance(operation, str) and operation in ['min', 'max', 'sum']:
            self.operation = lambda current_step_values, new_step_values: numexpr.evaluate(f'{operation}(data, axis=0)',
                                                local_dict={"data": [current_step_values, new_step_values]})
        else:
            self.operation = operation

    def in_window(self, step: int) -> bool:
        """
        Returns if step is in window interval
        """
        if self.include_init:
            return step >= self.start and step <= self.end
        return step > self.start and step <= self.end

    def add_step_values(self, step: int, step_values: np.array):
        """
        Adds contribution of data values for specified step, if inside window, by computing
        reduction operation on existing step values and new step values -
        saves on memory as only the reduction operation on processed steps
        is stored
        """
        if not self.in_window(step):
            return
        if len(self.step_values) == 0:
            self.step_values = step_values
        else:
            try:
                self.step_values = self.operation(self.step_values, step_values)
            except:
                raise TypeError(self.start, self.end)

    def reached_end_step(self, step: int) -> bool:
        """
        Returns if end step has been reached
        """
        return step == self.end

    def size(self) -> int:
        """
        Returns size of window interval
        """
        return self.end - self.start