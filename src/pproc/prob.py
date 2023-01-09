#!/usr/bin/env python3
import numpy as np
import numexpr
from typing import List, Dict
import yaml
import sys
import argparse
from datetime import datetime, timedelta

import pyfdb
import eccodes


def read_gribs(request, fdb, step, paramId) -> List[eccodes.GRIBMessage]:

    # Modify FDB request and read input data
    request["param"] = paramId
    request["step"] = step

    fdb_reader = fdb.retrieve(request)
    eccodes_reader = eccodes.StreamReader(fdb_reader)
    messages = list(eccodes_reader)
    assert len(messages) == len(request["number"])
    return messages


def grib_set(grib_file, key_values: Dict) -> None:
    """
    Sets the dictionary of keys and values into the grib file
    """
    for key, value in key_values.items():
        grib_file.set(key, value)


def grib_check(grib_file, key_values: Dict) -> None:
    """
    Checks the values of the specified keys have been set correctly in
    the grib file. Otherwise throws a ValueError.
    """
    for key, value in key_values.items():
        grib_value = grib_file.get(key)
        cast_value = value
        if not isinstance(value, type(grib_value)):
            # int values are returned as strings and floats as integers
            if isinstance(value, int):
                cast_value = str(value)
            elif isinstance(value, float):
                cast_value = int(value)
        if grib_value != cast_value:
            raise ValueError(
                f"GribCheck: key {key} expected value {cast_value}. Got {grib_value}.")


def write_instantaneous_grib(fdb, template_grib, step, threshold, data) -> None:

    # Copy an input GRIB message and modify headers for writing probability
    # field
    out_grib = template_grib.copy()
    key_values = {
        "step": step,
        "type": "ep",
        "paramId": threshold["out_paramid"],
        "localDefinitionNumber": 5,
        "localDecimalScaleFactor": 2,
        "thresholdIndicator": 2,
        "upperThreshold": threshold["value"]
    }
    grib_set(out_grib, key_values)

    # Set GRIB data and write to FDB
    out_grib.set_array("values", data)
    fdb.archive(out_grib.get_buffer())


def write_period_grib(fdb, template_grib, leg, start_step, end_step, threshold, data) -> None:

    # Copy an input GRIB message and modify headers for writing probability
    # field
    out_grib = template_grib.copy()
    key_values = {
        "type": "ep",
        "paramId": threshold["out_paramid"],
        "localDefinitionNumber": 5,
        "localDecimalScaleFactor": 2,
        "thresholdIndicator": 2,
        "upperThreshold": threshold["value"],
        "stepType": "max",
        "stepRange": f"{start_step}-{end_step}",
    }
    if leg == 2:
        key_values["unitOfTimeRange"] = 11

    grib_set(out_grib, key_values)

    # Set GRIB data and write to FDB
    out_grib.set_array("values", data)
    fdb.archive(out_grib.get_buffer())


def ensemble_probability(data: np.array, threshold) -> np.array:
    """ Ensemble Probabilities:

        Computes the probability of a given parameter crossing a given threshold,
        by checking how many times it occurs across all ensembles.
        e.g. the chance of temperature being less than 0C

    """

    # Read threshold configuration and compute probability
    comparison = threshold["comparison"]
    comp = numexpr.evaluate("data " + comparison +
                            str(threshold["value"]), local_dict={"data": data})
    probability = np.where(comp, 100, 0).mean(axis=0)

    return probability


class Window:

    """
    Class for collating data for all ensembles over an step interval for computing
    time-averaged probabilities
    """

    def __init__(self, start_step: int, end_step: int, operation: str):
        """
        :param start_step: start step of interval, not inclusive
        :param end_step: end_step of interval, inclusive
        :param operation: string for name of reduction operation e.g. min, max, sum
        """
        self.start_step = start_step
        self.end_step = end_step
        self.operation = operation
        self.step_values = []

    def in_window(self, step: int) -> bool:
        """
        Returns if step is in window interval
        """
        return step > self.start_step and step <= self.end_step

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
            self.step_values = numexpr.evaluate(f'{self.operation}(data, axis=0)',
                                                local_dict={"data": [self.step_values, step_values]})

    def reached_end_step(self, step: int) -> bool:
        """
        Returns if end step has been reached
        """
        return step == self.end_step

    def size(self) -> int:
        """
        Returns size of window interval
        """
        return self.end_step - self.start_step


class InstantaneousWindow(Window):

    def __init__(self, step: int):
        super().__init__(step - 1, step, 'min')


def main(args=None):

    parser = argparse.ArgumentParser(description='Compute instantaneous and period probabilities')
    parser.add_argument('-c', '--config', required=True, help='YAML configuration file')
    parser.add_argument('-d', '--date', required=True, help='Forecast date')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    date = datetime.strptime(args.date, "%Y%m%d%H")

    fdb = pyfdb.FDB()

    # Read base config
    leg = config.get("leg")
    nensembles = config.get("number_of_ensembles", 50)

    base_request = config["base_request"]
    base_request["number"] = range(1, nensembles)
    base_request['date'] = date.strftime("%Y%m%d")
    base_request['time'] = date.strftime("%H") + '00'

    thresholds = config["thresholds"]
    for threshold in thresholds:

        paramid = threshold["in_paramid"]

        # Sort steps and create instantaneous windows
        windows = []
        unique_steps = []
        for steps in config.get("steps"):
            start_step = steps['start_step']
            end_step = steps['end_step']
            interval = steps['interval']
            write = steps.get('write', False)
            for step in range(start_step, end_step + 1, interval):
                if step not in unique_steps:
                    unique_steps.append(step)
                    if write:
                        windows.append(InstantaneousWindow(step))

        # Create windows from periods
        for periods in config['periods']:
            windows.append(Window(periods['start_step'], periods['end_step'],
                                  periods['operation']))

        for step in sorted(unique_steps):
            messages = read_gribs(base_request, fdb, step, paramid)
            data = np.asarray([message.get_array('values')
                              for message in messages])

            new_windows = []
            for window in windows:
                window.add_step_values(step, data)

                # Write out probabilites if end of window has been reached
                if window.reached_end_step(step):
                    window_probability = ensemble_probability(
                        window.step_values, threshold)

                    if window.size() == 1:
                        print(f"Writing instantaneous probability for param {paramid} at step {step}")
                        write_instantaneous_grib(fdb, messages[0], step, threshold, window_probability)
                    else:
                        print(f"Writing time-averaged {window.start_step}-{window.end_step} probability for {paramid}")
                        write_period_grib(
                            fdb, messages[0], leg, window.start_step, window.end_step, threshold, window_probability)
                else:
                    new_windows.append(window)
            windows = new_windows

    fdb.flush()


if __name__ == "__main__":
    main(sys.argv)
