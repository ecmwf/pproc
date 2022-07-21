#!/usr/bin/env python3
import numpy as np
import numexpr
from typing import List
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


def write_instantaneous_grib(fdb, template_grib, step, threshold, data) -> None:

    # Copy an input GRIB message and modify headers for writing probability field
    out_grib = template_grib.copy()
    out_grib.set("step", step)
    out_grib.set("type", "ep")
    out_grib.set("paramId", threshold["out_paramid"])
    out_grib.set("localDefinitionNumber", 5)
    out_grib.set("localDecimalScaleFactor", 2)
    out_grib.set("thresholdIndicator", 2)
    out_grib.set("upperThreshold", threshold["value"])

    # Set GRIB data and write to FDB
    out_grib.set_array("values", data)
    fdb.archive(out_grib.get_buffer())


def write_period_grib(fdb, template_grib, leg, start_step, end_step, threshold, data) -> None:

    # Copy an input GRIB message and modify headers for writing probability field
    out_grib = template_grib.copy()
    out_grib.set("stepRange", f"{start_step}-{end_step}")
    out_grib.set("type", "ep")
    out_grib.set("paramId", threshold["out_paramid"])
    out_grib.set("stepType", "max")
    out_grib.set("localDefinitionNumber", 5)
    out_grib.set("localDecimalScaleFactor", 2)
    out_grib.set("thresholdIndicator", 2)
    out_grib.set("upperThreshold", threshold["value"])

    if leg == 2:
        out_grib.set("unitOfTimeRange", 11)

    # Set GRIB data and write to FDB
    out_grib.set_array("values", data)
    fdb.archive(out_grib.get_buffer())


def instantaneous_probability(data: np.array, threshold) -> np.array:

    """ Instantaneous Probabilities:

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
    base_request['time'] = date.strftime("%H")+'00'

    thresholds = config["thresholds"]
    for threshold in thresholds:

        paramid = threshold["in_paramid"]
        
        # Instantaneous probabilities
        # - Reads all ensembles for every given step and computes the probability of a threshold being reached
        probabilities = {}
        for steps in config.get("steps"):
            
            start_step = steps['start_step']
            end_step = steps['end_step']
            interval = steps['interval']
            write = steps.get('write', False)
            
            for step in range(start_step, end_step+1, interval):
                if step not in probabilities:
                    messages = read_gribs(base_request, fdb, step, paramid)
                    data = np.asarray([message.get_array('values') for message in messages])
                    probability = instantaneous_probability(data, threshold)
                    probabilities[step] = probability
                if write:
                    print(f"Writing instantaneous probability for param {paramid} at step {step}")
                    write_instantaneous_grib(fdb, messages[0], step, threshold, probabilities[step])
        
        all_steps = sorted(probabilities.keys())
        print(f'Total number of steps available:\n {all_steps}')

        # Time-averaged probabilities
        # - Takes the instantaneous probabilities computes the time average window
        for periods in config['periods']:
            start_step = periods['start_step']
            end_step = periods['end_step']

            mean_probability = 0
            period_size = 0
            for i, step in enumerate(all_steps):
                if step > start_step and step <= end_step:
                    print(step)
                    if i == 0:
                        dt = step
                    else:
                        dt = step - all_steps[i-1]
                    mean_probability += probabilities[step]/dt
                    period_size += dt
            mean_probability *= period_size

            if period_size != (end_step - start_step):
                raise Exception(f'Period size {period_size} does not match window from config')

            print(f"Writing time-averaged {start_step}-{end_step} probability for {paramid}")
            write_period_grib(fdb, messages[0], leg, start_step, end_step, threshold, mean_probability)

    fdb.flush()


if __name__ == "__main__":
    main(sys.argv)
