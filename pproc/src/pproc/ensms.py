#!/usr/bin/env python3

import numpy as np

from datetime import datetime  # , timedelta

import eccodes


def read_gribs(request, fdb, step, paramId):
    pass


#     # Modify FDB request and read input data
#     request["param"] = paramId
#     request["step"] = step

#     fdb_reader = fdb.retrieve(request)
#     eccodes_reader = eccodes.StreamReader(fdb_reader)
#     messages = list(eccodes_reader)
#     assert len(messages) == len(request["number"])
#     return messages


def write_instantaneous_grib(fdb, template_grib, step, threshold, data) -> None:
    pass


#     # Copy an input GRIB message and modify headers for writing probability field
#     out_grib = template_grib.copy()
#     out_grib.set("step", step)
#     out_grib.set("type", "ep")
#     out_grib.set("paramId", threshold["out_paramid"])
#     out_grib.set("localDefinitionNumber", 5)
#     out_grib.set("localDecimalScaleFactor", 2)
#     out_grib.set("thresholdIndicator", 2)
#     out_grib.set("upperThreshold", threshold["value"])

#     # Set GRIB data and write to FDB
#     out_grib.set_array("values", data)
#     fdb.archive(out_grib.get_buffer())


def write_period_grib(
    fdb, template_grib, leg, start_step, end_step, threshold, data
) -> None:
    pass


#     # Copy an input GRIB message and modify headers for writing probability field
#     out_grib = template_grib.copy()
#     out_grib.set("stepRange", f"{start_step}-{end_step}")
#     out_grib.set("type", "ep")
#     out_grib.set("paramId", threshold["out_paramid"])
#     out_grib.set("stepType", "max")
#     out_grib.set("localDefinitionNumber", 5)
#     out_grib.set("localDecimalScaleFactor", 2)
#     out_grib.set("thresholdIndicator", 2)
#     out_grib.set("upperThreshold", threshold["value"])

#     if leg == 2:
#         out_grib.set("unitOfTimeRange", 11)

#     # Set GRIB data and write to FDB
#     out_grib.set_array("values", data)
#     fdb.archive(out_grib.get_buffer())


def instantaneous_probability(messages, threshold) -> np.array:
    pass


#     """ Instantaneous Probabilities:

#         Computes the probability of a given parameter crossing a given threshold,
#         by checking how many times it occurs across all ensembles.
#         e.g. the chance of temperature being less than 0C

#     """

#     data = np.asarray([message.get_array('values') for message in messages])

#     # Read threshold configuration and compute probability
#     comparison = threshold["comparison"]
#     comp = numexpr.evaluate("data " + comparison + str(threshold["value"]), local_dict={"data": data})
#     probability = np.where(comp, 100, 0).mean(axis=0)

#     return probability


def parse_range(r: str):
    if not r:
        return ()

    l = r.lower().split("/")
    if len(l) == 5 and l[1] == "to" and l[3] == "by":
        return range(int(l[0]), int(l[2]) + 1, int(l[4]))
    if len(l) == 3 and l[1] == "to":
        return range(int(l[0]), int(l[2]) + 1, 1)
    return sorted(set(map(int, l)))


def mars(input: str = ""):
    from subprocess import run
    from os import environ

    p = run(
        "/Users/mapm/.local/bin/mars",
        capture_output=True,
        text=True,
        input=input,
        env=environ,
    )
    if p.returncode:
        print(f"stdout:\n{p.stdout}\n")
        print(f"stderr:\n{p.stderr}\n")

    return p.returncode == 0


def main(args=None):
    import argparse
    from os import path

    from pproc.Config import VariableTree

    # arguments
    parser = argparse.ArgumentParser(
        description="Calculate ensemble means and standard deviations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config-file", help="Configuration file", required=True)
    parser.add_argument("--config-node", help="Configuration node", nargs="+")

    parser.add_argument("--no-mars", action="store_false", dest="mars")

    parser.add_argument("--class", default="od", dest="klass")
    parser.add_argument("--stream", default="enfo")
    parser.add_argument("--expver", default=1)
    parser.add_argument("--date", required=True)
    parser.add_argument("--time", choices=(0, 6, 12, 18), default=0)
    parser.add_argument("--grid", default="O640")
    parser.add_argument("--number", default="1/to/50")
    args = parser.parse_args()

    date = datetime.strptime(args.date, "%Y%m%d")

    # variables
    tree = VariableTree(args.config_file)
    var = tree.variables(*map(lambda n: int(n) if n.isdigit() else n, args.config_node))
    var.update(vars(args))
    var["class"] = var.pop("klass")
    # print(var)

    target = (
        "param={param},"
        + ("levtype=sfc", "levelist={levelist}")[bool(var["levelist"])]
        + ",step={step}"
    )

    pf = {
        "_verb": "retrieve",
        "class": var["class"],
        "type": "pf",
        "stream": var["stream"],
        "param": var["param"],
        "date": var["date"],
        "time": var["time"],
        "levtype": var["levtype"],
        "levelist": var["levelist"],
        "resol": "av",
        "grid": var["grid"],
        "expver": var["expver"],
        "step": var["step"],
        "number": var["number"],
        "target": f"'{target}'",
    }

    cf = {"_verb": "retrieve", "type": "cf", "number": "off"}

    r = lambda d: ",".join(f"{k}={v}" for k, v in d.items() if v)[6:]
    request = f"""
    {r(pf)}
    {r(cf)}
    """
    print(request)

    if var["mars"]:
        assert mars(request)

    print(parse_range(var["step"]))
    print(parse_range(var["levelist"]))


#     thresholds = config.get("thresholds", [])
#     for threshold in thresholds:

#         paramid = threshold["in_paramid"]
#
#         # Instantaneous probabilities
#         # - Reads all ensembles for every given step and computes the probability of a threshold being reached
#         probabilities = {}
#         for steps in config.get("steps"):
#
#             start_step = steps['start_step']
#             end_step = steps['end_step']
#             interval = steps['interval']
#             write = steps.get('write', False)
#
#             for step in range(start_step, end_step+1, interval):
#                 if step not in probabilities:
#                     messages = read_gribs(base_request, fdb, step, paramid)
#                     probability = instantaneous_probability(messages, threshold)
#                     probabilities[step] = probability
#                 if write:
#                     print(f"Writing instantaneous probability for param {paramid} at step {step}")
#                     write_instantaneous_grib(fdb, messages[0], step, threshold, probabilities[step])
#
#         all_steps = sorted(probabilities.keys())
#         print(f'Total number of steps available:\n {all_steps}')

#         # Time-averaged probabilities
#         # - Takes the instantaneous probabilities computes the time average window
#         for window in config.get('windows'):
#             start_step = window['start_step']
#             end_step = window['end_step']

#             mean_probability = 0
#             window_size = 0
#             for i, step in enumerate(all_steps):
#                 if step > start_step and step <= end_step:
#                     print(step)
#                     if i == 0:
#                         dt = step
#                     else:
#                         dt = step - all_steps[i-1]
#                     mean_probability += probabilities[step]/dt
#                     window_size += dt
#             mean_probability *= window_size

#             if window_size != (end_step - start_step):
#                 raise Exception(f'window size {window_size} does not match window from config')

#             print(f"Writing time-averaged {start_step}-{end_step} probability for {paramid}")
#             write_period_grib(fdb, messages[0], leg, start_step, end_step, threshold, mean_probability)

#     fdb.flush()


if __name__ == "__main__":
    import sys

    main(sys.argv)
