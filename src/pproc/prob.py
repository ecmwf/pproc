#!/usr/bin/env python3
import argparse
import sys
from datetime import datetime
from typing import Dict, Tuple

import numexpr
import numpy as np
import yaml

import eccodes
import pyfdb

from pproc.common import WindowManager

MISSING_VALUE = 9999


def read_gribs(request, fdb, step, paramId) -> Tuple[eccodes.GRIBMessage, np.array]:

    # Modify FDB request and read input data
    request["param"] = paramId
    request["step"] = step

    fdb_reader = fdb.retrieve(request)
    eccodes_reader = eccodes.StreamReader(fdb_reader)
    messages = list(eccodes_reader)
    assert len(messages) == len(request["number"])

    data = np.asarray([message.get_array("values") for message in messages])
    # Replace missing values with nan
    data = np.where(data == MISSING_VALUE, np.nan, data)

    return messages[0], data


def threshold_grib_headers(threshold) -> Dict:
    threshold_dict = {"paramId": threshold["out_paramid"]}
    threshold_value = threshold["value"]
    if 'localDecimalScaleFactor' in threshold:
        scale_factor = threshold['localDecimalScaleFactor']
        threshold_dict["localDecimalScaleFactor"] = scale_factor
        threshold_value = round(threshold["value"] * 10**scale_factor, 0)

    comparison = threshold["comparison"]
    if "<" in comparison:
        threshold_dict.update(
            {"thresholdIndicator": 2, "upperThreshold": threshold_value}
        )
    elif ">" in comparison:
        threshold_dict.update(
            {"thresholdIndicator": 1, "lowerThreshold": threshold_value}
        )
    return threshold_dict


def write_grib(fdb, template_grib, window_grib_headers, threshold, data) -> None:

    # Copy an input GRIB message and modify headers for writing probability
    # field
    out_grib = template_grib.copy()
    key_values = {
        "type": "ep",
        "localDefinitionNumber": 5,
        "bitsPerValue": 8,  # Set equal to accuracy used in mars compute
    }
    key_values.update(window_grib_headers)
    key_values.update(threshold_grib_headers(threshold))

    out_grib.set(key_values, check_values=True)

    # Set GRIB data and write to FDB
    out_grib.set_array("values", data)
    fdb.archive(out_grib.get_buffer())


def ensemble_probability(data: np.array, threshold) -> np.array:
    """Ensemble Probabilities:

    Computes the probability of a given parameter crossing a given threshold,
    by checking how many times it occurs across all ensembles.
    e.g. the chance of temperature being less than 0C

    """

    # Find all locations where np.nan appears as an ensemble value
    is_nan = np.isnan(np.sum(data, axis=0))

    # Read threshold configuration and compute probability
    comparison = threshold["comparison"]
    comp = numexpr.evaluate(
        "data " + comparison + str(threshold["value"]), local_dict={"data": data}
    )
    probability = np.where(comp, 100, 0).mean(axis=0)

    # Put in missing values
    probability = np.where(is_nan, MISSING_VALUE, probability)

    return probability


def main(args=None):

    parser = argparse.ArgumentParser(
        description="Compute instantaneous and period probabilities"
    )
    parser.add_argument("-c", "--config", required=True, help="YAML configuration file")
    parser.add_argument("-d", "--date", required=True, help="Forecast date")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print(config)

    date = datetime.strptime(args.date, "%Y%m%d%H")

    fdb = pyfdb.FDB()

    # Read base config
    leg = config.get("leg")
    nensembles = config.get("number_of_ensembles", 50)

    parameters = config["parameters"]
    for parameter in parameters:
        base_request = parameter["base_request"]
        base_request["number"] = range(1, nensembles + 1)
        base_request["date"] = date.strftime("%Y%m%d")
        base_request["time"] = date.strftime("%H") + "00"

        paramid = parameter["in_paramid"]

        window_manager = WindowManager(parameter)

        for step in window_manager.unique_steps:
            message_template, data = read_gribs(base_request, fdb, step, paramid)
            
            if 'other_parameter' in parameter:
                for second_param in parameter['other_parameter']:
                    _, data2 = read_gribs(base_request, fdb, step, second_param['paramid'])
                    if second_param['combine_operation'] == 'norm':
                        data = np.linalg.norm([data, data2], axis=0)
                    else:
                        data = getattr(np, second_param['combine_operation'])([data, data2], axis=0)

            completed_windows = window_manager.update_windows(step, data)
            for window in completed_windows:
                for threshold in window.thresholds:
                    window_probability = ensemble_probability(
                        window.step_values, threshold
                    )

                    print(
                        f"Writing probability for input param {paramid} and output "
                        + f"param {threshold['out_paramid']} for step(s) {window.name}"
                    )
                    write_grib(
                        fdb,
                        message_template,
                        window.grib_header(leg),
                        threshold,
                        window_probability,
                    )

    fdb.flush()


if __name__ == "__main__":
    main(sys.argv)
