#!/usr/bin/env python3
import numpy as np
import numexpr
from typing import List
import yaml
import sys
import argparse
from datetime import datetime

import pyfdb
import eccodes
from common import WindowManager


def read_gribs(request, fdb, step, paramId) -> List[eccodes.GRIBMessage]:

    # Modify FDB request and read input data
    request["param"] = paramId
    request["step"] = step

    fdb_reader = fdb.retrieve(request)
    eccodes_reader = eccodes.StreamReader(fdb_reader)
    messages = list(eccodes_reader)
    assert len(messages) == len(request["number"])
    return messages


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
        "upperThreshold": threshold['value']
    }
    out_grib.set(key_values, check_values=True)

    # Set GRIB data and write to FDB
    out_grib.set_array("values", data)
    fdb.archive(out_grib.get_buffer())


def write_period_grib(fdb, template_grib, leg, start_step, end_step, threshold, data) -> None:

    # Copy an input GRIB message and modify headers for writing probability
    # field
    out_grib = template_grib.copy()
    key_values = {
        "type": "ep",
        "paramId": threshold['out_paramid'],
        "localDefinitionNumber": 5,
        "localDecimalScaleFactor": 2,
        "thresholdIndicator": 2,
        "upperThreshold": threshold['value'],
        "stepType": "max",
        "stepRange": f"{start_step}-{end_step}",
    }
    if leg == 2:
        key_values["unitOfTimeRange"] = 11

    out_grib.set(key_values, check_values=True)

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
    comp = numexpr.evaluate("data " + comparison + str(threshold["value"]), local_dict={"data": data})
    probability = np.where(comp, 100, 0).mean(axis=0)

    return probability

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

    parameters = config["parameters"]
    for parameter in parameters:
        paramid = parameter["in_paramid"]

        # Check all threshold comparisons are the same 
        thresholds = parameter['thresholds']
        for threshold in thresholds[1:]:
            if threshold['comparison'] != thresholds[0]['comparison']:
                print(f'Different comparison operations for thresholds is currently not supported. Skipping \
                parameter id {paramid}')
                continue

        window_manager = WindowManager(parameter)
                            
        for step in sorted(window_manager.unique_steps):
            messages = read_gribs(base_request, fdb, step, paramid)
            data = np.asarray([message.get_array('values')
                              for message in messages])

            completed_windows = window_manager.update_windows(step, data)
            for window in completed_windows:
                for threshold in thresholds:
                    window_probability = ensemble_probability(
                        window.step_values, threshold)

                    if window.size() == 0:
                        print(f"Writing instantaneous probability for param {paramid} at step {step}")
                        write_instantaneous_grib(fdb, messages[0], step, threshold,
                        window_probability)
                    else:
                        print(f"Writing time-averaged {window.name} probability for {paramid}")
                        write_period_grib(
                            fdb, messages[0], leg, window.start, window.end, threshold,
                            window_probability)

            if window_manager.windows_completed():
                break

    fdb.flush()


if __name__ == "__main__":
    main(sys.argv)
