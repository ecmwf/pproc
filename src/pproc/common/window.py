import numexpr
import numpy as np
from typing import Dict


class Window:
    """
    Class for collating data for all ensembles over an step interval
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
        if window_size == 0:
            self.name = self.end
        else:
            self.name = f"{self.start}-{self.end}"
        self.step_values = []
        self.config_grib_header = {}

    def operation(self, new_step_values: np.array):
        """
        Combines data from unprocessed steps with existing step data values,
        and updates step data values. Any processing involving NaN values 
        must return NaN to be compatible with MARS compute

        :param new_step_values: data from new step
        """
        raise NotImplementedError

    def __contains__(self, step: int) -> bool:
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
        reduction operation on existing step values and new step values - only the reduction
        operation on processed steps is stored

        :param step: step to update window with
        :param step_values: data values for step
        """
        if step not in self:
            return
        if len(self.step_values) == 0:
            self.step_values = step_values
        else:
            self.operation(step_values)

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

    def grib_header(self, leg: int) -> Dict:
        """
        Returns window specific grib headers, including headers defined in
        config file

        :param leg: model leg
        :return: dictionary of header keys and values
        """
        header = self.config_grib_header
        if isinstance(self.name, int):
            header["step"] = self.name
        else:
            header.update({"stepType": "max", "stepRange": self.name})
            if leg == 2:
                header["unitOfTimeRange"] = 11

        return header


class DiffWindow(Window):
    """
    Window with operation that takes difference between the end and start step. Only accepts data
    from these two steps
    """

    def __init__(self, window_options):
        super().__init__(window_options, include_init=True)

    def operation(self, new_step_values: np.array):
        """
        Combines data from unprocessed steps with existing step data values,
        and updates step data values. Any processing involving NaN values 
        must return NaN to be compatible with MARS compute

        :param new_step_values: data from new step
        """
        self.step_values = np.subtract(new_step_values, self.step_values)

    def __contains__(self, step: int) -> bool:
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
        self.operation_str = window_operation

    def operation(self, new_step_values: np.array):
        """
        Combines data from unprocessed steps with existing step data values,
        and updates step data values. Any processing involving NaN values 
        must return NaN to be compatible with MARS compute

        :param new_step_values: data from new step
        """
        self.step_values = numexpr.evaluate(
            f"{self.operation_str}(data, axis=0)",
            local_dict={"data": [
                np.where(np.isnan(new_step_values), np.nan, self.step_values), 
                np.where(np.isnan(self.step_values), np.nan, new_step_values)
            ]},
        )

class WeightedSumWindow(SimpleOpWindow):
    """
    Window with weighted sum operation. Weighted sum is computed by weighting the
    data for each step by the step duration, and then dividing the sum by the total duration of the
    window.
    """

    def __init__(self, window_options):
        """
        :param window_options: config specifying start and end of window
        """
        super().__init__(window_options, "sum")
        self.previous_step = self.start

    def add_step_values(self, step: int, step_values: np.array):
        """
        Adds the contributions data_i * dt_i where data_i and dt_i is the data and step duration for step i,
        if step i is in the window. When final step has been reached, divides sum by the total duration of
        window.

        :param step: step to update window with
        :param step_values: data values for step
        """
        if step not in self:
            return
        step_duration = step - self.previous_step
        if len(self.step_values) == 0:
            self.step_values = step_values * step_duration
        else:
            self.operation(step_values * step_duration)

        self.previous_step = step
        if self.reached_end_step(step):
            self.step_values = self.step_values / self.size()
