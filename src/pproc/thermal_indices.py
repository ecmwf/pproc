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
import logging
import os
import sys
from typing import List, Set
import signal
import functools
from datetime import datetime

import earthkit.data
import numpy as np
import psutil
import thermofeel as thermofeel
from meters import ResourceMeter, metered

from pproc.common import Config, WindowManager, default_parser, Recovery, parallel
from pproc.common.io import target_from_location
from pproc.common.parallel import (
    SynchronousExecutor,
    QueueingExecutor,
)
from pproc.thermo import helpers
from pproc.thermo.indices import ComputeIndices
from pproc.thermo.wrappers import ArrayFieldList

logging.getLogger("pproc").setLevel(os.environ.get("PPROC_LOGLEVEL", "INFO").upper())
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
__version__ = "2.0.0"


@metered("Process step", out=logger.info)
def process_step(args, config, window_id, fields, recovery):
    helpers.check_field_sizes(fields)
    basetime, validtime = helpers.get_datetime(fields)
    step = helpers.get_step(fields)
    time = basetime.hour

    logger.info(
        f"Compute indices step {step}, validtime {validtime.isoformat()} - "
        + f"basetime {basetime.date().isoformat()}, time {time}"
    )
    logger.debug(f"Inputs \n {fields.ls(namespace='mars')}")
    indices = ComputeIndices(config.out_keys)

    # Windspeed - shortName ws
    if config.is_target_param("intermediate", {"ws", "10si"}):
        ws = indices.calc_field("10si", indices.calc_ws, fields)
        helpers.write(config.target("intermediate"), ws)

    # Cosine of Solar Zenith Angle - shortName uvcossza - ECMWF product
    # TODO: 214001 only exists for GRIB1 -- but here we use it for GRIB2 (waiting for WMO)
    if config.is_target_param("intermediate", {"cossza", "uvcossza"}):
        cossza = indices.calc_field("uvcossza", indices.calc_cossza_int, fields)
        helpers.write(config.target("intermediate"), cossza)

    # direct solar radiation - shortName dsrp - ECMWF product
    if config.is_target_param("intermediate", {"dsrp"}):
        dsrp = indices.calc_field("dsrp", indices.approximate_dsrp, fields)
        helpers.write(config.target("intermediate"), dsrp)

    # Mean Radiant Temperature - shortName mrt - ECMWF product
    if config.is_target_param("indices", {"mrt"}):
        mrt = indices.calc_field("mrt", indices.calc_mrt, fields)
        helpers.write(config.target("indices"), mrt)

    # Univeral Thermal Climate Index - shortName utci - ECMWF product
    if config.is_target_param("indices", {"utci"}):
        utci = indices.calc_field(
            "utci",
            indices.calc_utci,
            fields,
            print_misses=args.utci_misses,
            validate=args.validateutci,
        )
        helpers.write(config.target("indices"), utci)

    # Heat Index (adjusted) - shortName heatx - ECMWF product
    if config.is_target_param("indices", {"heatx"}):
        heatx = indices.calc_field("heatx", indices.calc_heatx, fields)
        helpers.write(config.target("indices"), heatx)

    # Wind Chill factor - shortName wcf - ECMWF product
    if config.is_target_param("indices", {"wcf"}):
        wcf = indices.calc_field("wcf", indices.calc_wcf, fields)
        helpers.write(config.target("indices"), wcf)

    # Apparent Temperature - shortName aptmp - ECMWF product
    if config.is_target_param("indices", {"aptmp"}):
        aptmp = indices.calc_field("aptmp", indices.calc_aptmp, fields)
        helpers.write(config.target("indices"), aptmp)

    # Relative humidity percent at 2m - shortName 2r - ECMWF product
    if config.is_target_param("indices", {"rhp", "2r"}):
        rhp = indices.calc_field("2r", indices.calc_rhp, fields)
        helpers.write(config.target("indices"), rhp)

    # Humidex - shortName hmdx
    if config.is_target_param("indices", {"hmdx"}):
        hmdx = indices.calc_field("hmdx", indices.calc_hmdx, fields)
        helpers.write(config.target("indices"), hmdx)

    # Normal Effective Temperature - shortName nefft
    if config.is_target_param("indices", {"nefft"}):
        nefft = indices.calc_field("nefft", indices.calc_nefft, fields)
        helpers.write(config.target("indices"), nefft)

    # Globe Temperature - shortName gt
    if config.is_target_param("indices", {"gt"}):
        gt = indices.calc_field("gt", indices.calc_gt, fields)
        helpers.write(config.target("indices"), gt)

    # Wet-bulb potential temperature - shortName wbpt
    if config.is_target_param("indices", {"wbpt"}):
        wbpt = indices.calc_field("wbpt", indices.calc_wbpt, fields)
        helpers.write(config.target("indices"), wbpt)

    # Wet Bulb Globe Temperature - shortName wbgt
    if config.is_target_param("indices", {"wbgt"}):  #
        wbgt = indices.calc_field("wbgt", indices.calc_wbgt, fields)
        helpers.write(config.target("indices"), wbgt)

    # effective temperature 261017
    # standard effective temperature 261019

    config.flush_targets()
    recovery.add_checkpoint(window_id)

    if args.usage:
        print_usage()


class ThermoConfig(Config):
    def __init__(self, args: argparse.Namespace, verbose: bool = True):
        super().__init__(args, verbose=verbose)

        self.date = datetime.strptime(str(self.options["fc_date"]), "%Y%m%d%H")
        self.out_keys = self.options.get("out_keys", {})
        self.sources = self.options.get("sources", {})
        self.root_dir = self.options.get("root_dir", None)
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)
        self._parse_windows()
        self.targets = {}
        for param_type, target_options in self.options.get("targets", {}).items():
            if param_type not in ["indices", "intermediate", "accum"]:
                raise ValueError(
                    f"Unknown target type {param_type}. Must be indices, intermediate or accum"
                )
            target_options = self.options.get("targets", {}).get(param_type, {})
            target = target_from_location(
                target_options.get("target", "null:"), overrides=self.override_output
            )
            if self.n_par_compute > 1:
                target.enable_parallel(parallel)
            if args.recover:
                target.enable_recovery()
            self.targets[param_type] = {
                "params": set(target_options.get("params", [])),
                "target": target,
            }

    def target(self, param_type: str):
        return self.targets[param_type]["target"]

    def is_target_param(self, param_type: str, valid_names: Set[str]) -> bool:
        return bool(self.targets[param_type]["params"] & valid_names.union({"all"}))

    def flush_targets(self):
        for param_target in self.targets.values():
            param_target["target"].flush()

    def _parse_windows(self):
        window_config = self.options.pop("windows")
        periods = []
        for roptions in window_config["ranges"]:
            periods += [
                {"range": [x, x + roptions["interval"]]}
                for x in range(
                    roptions["start_step"],
                    roptions["end_step"] + 1,
                    roptions["interval"],
                )
            ]
        self.options["windows"] = [
            {
                "window_operation": window_config["operation"],
                "periods": periods,
            }
        ]


def load_input(source: str, config: ThermoConfig, step: int):
    req = config.sources.get(source, {}).copy()
    if len(req) == 0:
        return None
    req.update(config.override_input)
    req["step"] = [step]

    src = req.pop("source")
    if ":" in src:
        src, loc = src.split(":")
    if src == "fdb":
        ds = earthkit.data.from_source("fdb", req, stream=True, read_all=True)
    elif src == "fileset":
        loc.format_map(req)
        req["paramId"] = req.pop("param")
        ds = earthkit.data.from_source("file", loc).sel(req)
    elif src == "mars":
        ds = earthkit.data.from_source("mars", req)
    else:
        raise ValueError(f"Unknown source {source}")

    if len(ds) == 0:
        raise ValueError(f"No data found for request {req} from source {source}")
    return earthkit.data.FieldList.from_array(ds.values, ds.metadata())


def get_parser():
    parser = default_parser("Compute thermal indices")

    parser.add_argument(
        "-a",
        "--accelerate",
        help="accelerate computations using JAX JIT",
        action="store_true",
    )
    parser.add_argument(
        "--validateutci",
        help="validate utci by detecting nans and out of bounds values. NOT to use in production. Very verbose option.",
        action="store_true",
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
    logger.info(
        f"Usage: cpu load {cpu_usage:5.1f}% -- proc mem {procmem:3.1f}GiB {procmemperc:3.1f}%"
        + f" -- sys mem {sysmem:3.1f}GiB {sysmemperc}%"
    )


def main(args: List[str] = sys.argv[1:]):
    sys.stdout.reconfigure(line_buffering=True)
    parser = get_parser()
    args = parser.parse_args(args)
    config = ThermoConfig(args)
    recovery = Recovery(config.root_dir, args.config, config.date, args.recover)
    last_checkpoint = recovery.last_checkpoint()
    executor = (
        SynchronousExecutor()
        if config.n_par_compute == 1
        else QueueingExecutor(
            config.n_par_compute,
            config.window_queue_size,
            initializer=signal.signal,
            initargs=(signal.SIGTERM, signal.SIG_DFL),
        )
    )

    logger.info(f"Compute Thermal Indices: {__version__}")
    logger.info(f"thermofeel: {thermofeel.__version__}")
    logger.info(f"earthkit.data: {earthkit.data.__version__}")
    logger.info(f"Numpy: {np.version.version}")
    logger.info(f"Python: {sys.version}")
    logger.debug(
        f"Parallel processes: {config.n_par_compute}, queue size: {config.window_queue_size}"
    )

    window_manager = WindowManager(config.options, {})
    if last_checkpoint is not None:
        checkpointed_windows = [
            recovery.checkpoint_identifiers(x)[0] for x in recovery.checkpoints
        ]
        window_manager.delete_windows(checkpointed_windows)
        logger.info(f"Recovery: looping from step {window_manager.unique_steps[0]}")
    thermo_partial = functools.partial(process_step, args, config, recovery=recovery)
    with executor:
        for step in window_manager.unique_steps:
            accum_data = load_input("accum", config, step)
            completed_windows = window_manager.update_windows(
                step, [] if accum_data is None else accum_data.values
            )
            for window_id, window in completed_windows:
                if window.size() == 0:
                    fields = load_input("inst", config, step)
                else:
                    # Set step range for de-accumulated fields
                    fields = earthkit.data.FieldList.from_array(
                        window.step_values,
                        [
                            x.override(stepType="diff", stepRange=window.name)
                            for x in accum_data.metadata()
                        ],
                    )
                    for field in fields:
                        metadata = field.metadata()
                        if config.is_target_param(
                            "accum",
                            {metadata.get("shortName"), metadata.get("paramId")},
                        ):
                            helpers.write(config.target("accum"), field)
                    config.target("accum").flush()

                fields += load_input("inst", config, step)

                executor.submit(
                    thermo_partial,
                    window_id,
                    ArrayFieldList(fields.values, fields.metadata()),
                )

        executor.wait()

    if args.timers:
        print_timers()

    return 0


if __name__ == "__main__":
    sys.exit(main())
