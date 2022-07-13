#!/usr/bin/env python3
import pyfdb
import eccodes
import numpy as np
import numexpr
from typing import List
import yaml
import sys
import argparse


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


def instantaneous_probability(messages: List[eccodes.GRIBMessage], threshold):

    """ Instantaneous Probabilities:

        Computes the probability of a given parameter crossing a given threshold,
        by checking how many times it occurs across all ensembles.
        e.g. the chance of temperature being less than 0C

    """

    data = np.asarray([message.get_array('values') for message in messages])

    # Read threshold configuration and compute probability
    comparison = threshold["comparison"]
    comp = numexpr.evaluate("data " + comparison + str(threshold["value"]), local_dict={"data": data})
    probability = np.where(comp, 100, 0).mean(axis=0)

    return probability


def main(args=None):

    parser = argparse.ArgumentParser(description='Compute instantaneous and period probabilities')
    parser.add_argument('-c', '--config', required=True, help='YAML configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    fdb = pyfdb.FDB()

    # Read base config
    leg = config.get("leg")
    nensembles = config.get("number_of_ensembles", 50)

    base_request = config.get("base_request")
    base_request["number"] = range(1, nensembles)

    thresholds = config.get("thresholds", [])

    # Instantaneous probabilities
    # - Reads all ensembles for every given step and computes the probability of a threshold being reached

    start_step = config.get("instantaneous", {}).get("start_step", 12)
    end_step = config.get("instantaneous", {}).get("end_step", 240)
    interval = config.get("instantaneous", {}).get("interval", 12)

    for threshold in thresholds:
        for step in range(start_step, end_step + 1, interval):
            messages = read_gribs(base_request, fdb, step, threshold["in_paramid"])
            probability = instantaneous_probability(messages, threshold)
            print(f"Writing instantaneous probability for param {threshold['in_paramid']} at step {step}")
            write_instantaneous_grib(fdb, messages[0], step, threshold, probability)

    # Period probabilities
    # - Takes the instantaneous probabilities at all 3h and 6h output steps and computes the overall mean

    start_step = config.get("period", {}).get("start_step", 120)
    end_step = config.get("period", {}).get("end_step", 240)
    h3_stop = config.get("period", {}).get("3hourly_output_cutoff_step", 144)

    # 3-hourly output steps (may be no steps if start_step > h3_stop)
    h3_steps = list(range(start_step + 3, min(h3_stop, end_step + 1), 3))
    # 6-hourly output steps (may be no steps if end_step < h3_stop)
    h6_steps = list(range(max(h3_stop, start_step), end_step + 1, 6))

    steps = h3_steps + h6_steps

    for threshold in thresholds:

        probabilities_accumulated = []

        for step in steps:
            messages = read_gribs(base_request, fdb, step, threshold["in_paramid"])
            probability = instantaneous_probability(messages, threshold)
            probabilities_accumulated.append(probability)
            print(f"Accumulating period probability for {threshold['in_paramid']} at step {step}")

        probabilities_accumulated = np.array(probabilities_accumulated)

        period_probability = probabilities_accumulated.mean(axis=0)

        print(f"Writing period probability for {threshold['in_paramid']}")
        write_period_grib(fdb, messages[0], leg, start_step, end_step, threshold, period_probability)

    fdb.flush()


if __name__ == "__main__":
    main(sys.argv)
