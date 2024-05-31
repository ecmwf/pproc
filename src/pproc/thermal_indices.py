#!/usr/bin/env python3

# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Note: This script is intended only as example usage of thermofeel library.
#       It is designed to be used with ECMWF forecast data.

import argparse
import os
import sys
from typing import List

import earthkit.data
import numpy as np
import psutil
import thermofeel as thermofeel
from codetiming import Timer

from pproc.common import Config, WindowManager, default_parser
from pproc.common.io import target_from_location
from pproc.thermo import helpers
from pproc.thermo.indices import ComputeIndices

__version__ = "2.0.0"


@Timer(name="proc_step", logger=None)
def process_step(args, config, step, fields, target):

    helpers.check_field_sizes(fields)
    basetime, validtime = helpers.get_datetime(fields)

    time = basetime.hour
    print(
        f"validtime {validtime.isoformat()} - basetime {basetime.date().isoformat()} : time {time} step {step}"
    )

    indices = ComputeIndices(config.out_keys)

    # Windspeed - shortName ws
    if args.ws:
        ws = indices.calc_field("ws", indices.calc_ws, fields)
        helpers.write(target, ws)

    # Cosine of Solar Zenith Angle - shortName uvcossza - ECMWF product
    # TODO: 214001 only exists for GRIB1 -- but here we use it for GRIB2 (waiting for WMO)
    if args.cossza:
        cossza = indices.calc_field("cossza", indices.calc_cossza_int, fields)
        helpers.write(target, cossza)

    # direct solar radiation - shortName dsrp - ECMWF product
    if args.dsrp:
        dsrp = indices.calc_field("dsrp", indices.approximate_dsrp, fields)
        helpers.write(target, dsrp)

    # Mean Radiant Temperature - shortName mrt - ECMWF product
    if args.mrt or args.all:
        mrt = indices.calc_field("mrt", indices.calc_mrt, fields)
        helpers.write(target, mrt)

    # Univeral Thermal Climate Index - shortName utci - ECMWF product
    if args.utci or args.all:
        utci = indices.calc_field(
            "utci",
            indices.calc_utci,
            fields,
            print_misses=args.utci_misses,
            validate=args.validateutci,
        )
        helpers.write(target, utci)

    # Heat Index (adjusted) - shortName heatx - ECMWF product
    if args.heatx or args.all:
        heatx = indices.calc_field("heatx", indices.calc_heatx, fields)
        helpers.write(target, heatx)

    # Wind Chill factor - shortName wcf - ECMWF product
    if args.wcf or args.all:
        wcf = indices.calc_field("wcf", indices.calc_wcf, fields)
        helpers.write(target, wcf)

    # Apparent Temperature - shortName aptmp - ECMWF product
    if args.aptmp or args.all:
        aptmp = indices.calc_field("aptmp", indices.calc_aptmp, fields)
        helpers.write(target, aptmp)

    # Relative humidity percent at 2m - shortName 2r - ECMWF product
    if args.rhp or args.all:
        rhp = indices.calc_field("rhp", indices.calc_rhp, fields)
        helpers.write(target, rhp)

    # Humidex - shortName hmdx
    if args.hmdx or args.all:
        hmdx = indices.calc_field("hmdx", indices.calc_hmdx, fields)
        helpers.write(target, hmdx)

    # Normal Effective Temperature - shortName nefft
    if args.nefft or args.all:
        nefft = indices.calc_field("nefft", indices.calc_nefft, fields)
        helpers.write(target, nefft)

    # Globe Temperature - shortName gt
    if args.gt or args.all:
        gt = indices.calc_field("gt", indices.calc_gt, fields)
        helpers.write(target, gt)

    # Wet-bulb potential temperature - shortName wbpt
    if args.wbpt or args.all:
        wbpt = indices.calc_field("wbpt", indices.calc_wbpt, fields)
        helpers.write(target, wbpt)

    # Wet Bulb Globe Temperature - shortName wbgt
    if args.wbgt or args.all:  #
        wbgt = indices.calc_field("wbgt", indices.calc_wbgt, fields)
        helpers.write(target, wbgt)

    # effective temperature 261017
    # standard effective temperature 261019

    target.flush()


class ThermoConfig(Config):
    def __init__(self, args: argparse.Namespace, verbose: bool = True):
        super().__init__(args, verbose=verbose)

        self.out_keys = self.options.get("out_keys", {})
        self.sources = self.options.get("sources", {})
        self.root_dir = self.options.get("root_dir", None)


def load_input(source: str, config: ThermoConfig, step: int):
    src, param_type = source.split(":")
    if src == "null":
        return None
    req = config.sources[src][param_type].copy()
    req.update(config.override_input)
    req["step"] = [step]
    if src == "fdb":
        ds = earthkit.data.from_source("fdb", req, stream=True, batch_size=0)
    elif src == "fileset":
        loc = req.pop("location")
        loc.format_map(req)
        req["paramId"] = req.pop("param")
        ds = earthkit.data.from_source("file", loc).sel(req)
    else:
        raise ValueError(f"Unknown source {source}")

    if len(ds) == 0:
        raise ValueError(f"No data found for request {req} from source {source}")
    return earthkit.data.FieldList.from_numpy(ds.values, ds.metadata())


def get_parser():
    parser = default_parser("Compute thermal indices")

    parser.add_argument(
        "-a",
        "--accelerate",
        help="accelerate computations using JAX JIT",
        action="store_true",
    )

    parser.add_argument(
        "--in-accum",
        required=True,
        type=str,
        help="Input source for accumulated parameters",
    )
    parser.add_argument(
        "--in-inst",
        required=True,
        type=str,
        help="Input source for instantaneous parameters",
    )
    parser.add_argument(
        "--out-indices", required=True, type=str, help="Target for computed indices"
    )

    parser.add_argument(
        "--all", help="compute all available indices", action="store_true"
    )

    parser.add_argument(
        "--validateutci",
        help="validate utci by detecting nans and out of bounds values. NOT to use in production. Very verbose option.",
        action="store_true",
    )

    parser.add_argument(
        "--ws", help="compute wind speed from components", action="store_true"
    )
    parser.add_argument(
        "--cossza",
        help="compute Cosine of Solar Zenith Angle (cossza)",
        action="store_true",
    )
    parser.add_argument(
        "--dsrp",
        help="compute dsrp (approximated)",
        action="store_true",
    )
    parser.add_argument("--mrt", help="compute mrt", action="store_true")
    parser.add_argument(
        "--utci",
        help="compute UTCI Universal Thermal Climate Index",
        action="store_true",
    )
    parser.add_argument(
        "--heatx", help="compute Heat Index (adjusted)", action="store_true"
    )
    parser.add_argument("--wcf", help="compute wcf factor", action="store_true")
    parser.add_argument(
        "--aptmp", help="compute Apparent Temperature", action="store_true"
    )
    parser.add_argument(
        "--rhp", help="compute relative humidity percent", action="store_true"
    )

    parser.add_argument("--hmdx", help="compute humidex", action="store_true")
    parser.add_argument(
        "--nefft", help="compute net effective temperature", action="store_true"
    )

    # TODO: these outputs are not yet in WMO GRIB2 recognised parameters
    parser.add_argument(
        "--wbgt", help="compute Wet Bulb Globe Temperature", action="store_true"
    )
    parser.add_argument("--gt", help="compute  Globe Temperature", action="store_true")
    parser.add_argument(
        "--wbpt", help="compute Wet Bulb Temperature", action="store_true"
    )

    parser.add_argument(
        "--timers",
        help="print function performance timers at the end",
        action="store_true",
    )
    parser.add_argument(
        "--usage", help="print cpu and memory usage during run", action="store_true"
    )
    parser.add_argument(
        "--utci-misses", help="print missing values for UTCI", action="store_true"
    )

    return parser


def print_timers():
    print("Performance summary:")
    print("--------------------")
    for t in Timer.timers.items():
        func, stats = t
        count = Timer.timers.count(func)
        if count > 0:
            mean = Timer.timers.mean(func)
            tmin = Timer.timers.min(func)
            tmax = Timer.timers.max(func)
            stdev = Timer.timers.stdev(func) if count > 1 else 0.0
            print(
                f"{func:<10} calls {count:>4}  --  avg + stdev [{mean:>8.4f} , {stdev:>8.4f}]s"
                + f" --  min + max [{tmin:>8.4f} , {tmax:>8.4f}] s"
            )


def print_usage():
    load1, load5, load15 = psutil.getloadavg()
    cpu_usage = (load5 / os.cpu_count()) * 100
    sysmem = psutil.virtual_memory().used / 1024**3  # in GiB
    sysmemperc = psutil.virtual_memory().percent
    procmem = psutil.Process(os.getpid()).memory_info().rss / 1024**3  # in GiB
    procmemperc = psutil.Process(os.getpid()).memory_percent()
    print(
        f"[INFO] usage: cpu load {cpu_usage:5.1f}% -- proc mem {procmem:3.1f}GiB {procmemperc:3.1f}%"
        + f" -- sys mem {sysmem:3.1f}GiB {sysmemperc}%"
    )


def main(args: List[str] = sys.argv[1:]):
    sys.stdout.reconfigure(line_buffering=True)
    parser = get_parser()
    args = parser.parse_args(args)
    config = ThermoConfig(args)
    target = target_from_location(args.out_indices, overrides=config.override_output)

    print(f"Compute Thermal Indices: {__version__}")
    print(f"thermofeel: {thermofeel.__version__}")
    print(f"earthkit.data: {earthkit.data.__version__}")
    print(f"Numpy: {np.version.version}")
    print(f"Python: {sys.version}")

    print("----------------------------------------")

    window_manager = WindowManager(config.options, {})
    for step in window_manager.unique_steps:
        accum_data = load_input(args.in_accum, config, step)
        completed_windows = window_manager.update_windows(
            step, [] if accum_data is None else accum_data.values
        )
        for _, window in completed_windows:
            if window.size() == 0:
                fields = load_input(args.in_inst, config, step)
            else:
                # Set step range for de-accumulated fields
                fields = earthkit.data.FieldList.from_numpy(
                    window.step_values,
                    [
                        x.override(stepType="diff", stepRange=window.name)
                        for x in accum_data.metadata()
                    ],
                )
                fields += load_input(args.in_inst, config, step)
            print(f"Step {step}, Input:")
            print(fields.ls(namespace="mars"))

            process_step(args, config, step, fields, target)

            if args.usage:
                print_usage()

            print("----------------------------------------")

    if args.timers:
        print_timers()

    return 0


if __name__ == "__main__":
    sys.exit(main())
