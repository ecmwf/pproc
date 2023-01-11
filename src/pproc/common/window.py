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
        self.start = int(window_options["range"][0])
        self.end = int(window_options["range"][1])
        self.include_init = include_init
        window_size = self.end - self.start
        self.suffix = f"{window_size:0>3}_{self.start:0>3}h_{self.end:0>3}h"
        self.name = f"{self.start}-{self.end}"
        self.operation = None
        self.step_values = []

    def set_operation(self, operation):
        """
        Sets reduction operation, if none, on existing step data values and new step data values.

        :param operation: function of the form f(current_step_values, new_step_values) -> reduced_step_values
        """
        if self.operation is None:
            self.operation = operation

    def in_window(self, step: int) -> bool:
        """
        :param step: current step
        :return: boolean specifying if step is in window interval
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

        :param step: step to update window with
        :param step_values: data values for step
        """
        if not self.in_window(step):
            return
        if len(self.step_values) == 0:
            self.step_values = step_values
        else:
            self.step_values = self.operation(self.step_values, step_values)

    def reached_end_step(self, step: int) -> bool:
        """
        :param step: current step
        :return: boolean specifying if current step is equal to window end step
        """
        return step == self.end

    def size(self) -> int:
        """
        :return: size of window interval
        """
        return self.end - self.start


class DiffWindow(Window):
    """
    Window with operation that takes difference between the end and start step. Only accepts data
    from these two steps
    """

    def __init__(self, window_options):
        super().__init__(window_options, include_init=True)
        self.set_operation(
            lambda current_step_values, new_step_values: new_step_values
            - current_step_values
        )

    def in_window(self, step: int) -> bool:
        """
        :param step: current step
        :return: boolean specifying if step is in window interval
        """
        return step == self.start or step == self.end


class SimpleOpWindow(Window):
    """
    Window with operation min, max or sum - reduction operations supported by numexpr
    """

    def __init__(
        self, window_options, window_operation: str, include_init: bool = False
    ):
        """
        :param window_options: config specifying start and end of window
        :param window_operation: name of reduction operation out of min, max, sum
        :param include_init: boolean specifying whether to include start step
        """
        super().__init__(window_options, include_init)
        self.set_operation(
            lambda current_step_values, new_step_values: numexpr.evaluate(
                f"{window_operation}(data, axis=0)",
                local_dict={"data": [current_step_values, new_step_values]},
            )
        )


class WeightedSumWindow(Window):
    """
    Window with weighted sum operation. Weighted sums is computed by weighting the
    data for each step by the step duration, and then dividing by the total duration of the
    window.
    """

    def __init__(self, window_options):
        """
        :param window_options: config specifying start and end of window
        """
        super().__init__(window_options, include_init=False)
        self.set_operation(
            lambda current_step_values, new_step_values: numexpr.evaluate(
                "sum(data, axis=0)",
                local_dict={"data": [current_step_values, new_step_values]},
            )
        )
        self.previous_step = 0

    def add_step_values(self, step: int, step_values: np.array):
        """
        Adds the contributions data_i * dt_i where data_i and dt_i is the data and step duration for step i,
        if step i is in the window. When final step has been reached, divides sum by the total duration of
        window.

        :param step: step to update window with
        :param step_values: data values for step
        """
        if not self.in_window(step):
            return
        if len(self.step_values) == 0:
            step_duration = step - self.start
            self.step_values = step_values * step_duration
        else:
            step_duration = step - self.previous_step
            self.step_values = self.operation(
                self.step_values, step_values * step_duration
            )

        self.previous_step = step
        if self.reached_end_step(step):
            self.step_values = self.step_values / (self.end - self.start)
