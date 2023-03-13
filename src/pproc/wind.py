#!/usr/bin/env python3
#
# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.
import os
import sys
from io import BytesIO
from datetime import datetime
import numpy as np
import xarray as xr

import eccodes
import pyfdb
import mir

from pproc import common


def retrieve_messages(cfg, req, cached_file):
    if cfg.vod2uv:
        print(req)
        common.fdb_read_to_file(cfg.fdb, req, cached_file)
        messages = mir_wind(cfg, cached_file)
    else:
        out = common.fdb_retrieve(cfg.fdb, req, cfg.interpolation_keys)
        reader = eccodes.StreamReader(out)
        if not reader.peek():
            raise RuntimeError(f'No data retrieved for request {req}')
        messages = list(reader)
    return messages


def fdb_request_det(cfg, levelist, steps, name):
    """
    Retrieve vorticity and divergence or u/v for deterministic forecast
    """

    req = cfg.request.copy()
    req.pop("stream_ens")
    req["stream"] = req.pop("stream_det")
    req["date"] = cfg.date.strftime("%Y%m%d")
    req["time"] = cfg.date.strftime("%H") + "00"
    if req["levtype"] != "sfc":
        req["levelist"] = levelist
    req["step"] = steps
    req["type"] = "fc"

    return retrieve_messages(cfg, req, f"wind_det_{levelist}_{name}.grb")


def fdb_request_ens(cfg, levelist, steps, name):
    """
    Retrieve vorticity and divergence or u/v for ensemble forecast
    (control + perturbed)
    """

    req = cfg.request.copy()
    req.pop("stream_det")
    req["stream"] = req.pop("stream_ens")
    req["date"] = cfg.date.strftime("%Y%m%d")
    req["time"] = cfg.date.strftime("%H") + "00"
    if req["levtype"] != "sfc":
        req["levelist"] = levelist
    req["step"] = steps

    req_cf = req.copy()
    req_cf["type"] = "cf"
    messages = retrieve_messages(cfg, req_cf, f"wind_det_{levelist}_{name}.grb")

    req_pf = req.copy()
    req_pf["type"] = "pf"
    req_pf["number"] = range(1, cfg.members + 1)
    messages += retrieve_messages(cfg, req_pf, f"wind_det_{levelist}_{name}.grb")

    return messages


def mir_wind(cfg, cached_file):
    """
    Compute wind components from cached grib file
    The grib file contains the vorticity and the divergence
    returns a list of messages containing the two components of velocity
    """

    interp_keys = cfg.interpolation_keys

    out = BytesIO()
    inp = mir.MultiDimensionalGribFileInput(cached_file, 2)

    job = mir.Job(vod2uv="1", **interp_keys)
    job.execute(inp, out)

    out.seek(0)
    reader = eccodes.StreamReader(out)
    messages = list(reader)

    return messages


def wind_speed(messages):
    """
    Compute wind speed from grib messages containing u and v
    """
    steps = list(set([m["step"] for m in messages]))
    wind_paramids = list(set([m["paramId"] for m in messages]))
    assert len(wind_paramids) == 2

    u = {step: [] for step in steps}
    v = {step: [] for step in steps}
    for m in messages:
        step = m["step"]
        param = m["paramId"]
        if param == wind_paramids[0]:
            u[step].append(m.get_array("values"))
        elif param == wind_paramids[1]:
            v[step].append(m.get_array("values"))
        else:
            raise ValueError(f"Wrong paramId in message: {param}")

    u = np.asarray(list(u.values()))
    v = np.asarray(list(v.values()))

    ws = np.sqrt(u * u + v * v)
    ws = dict(zip(steps, ws))

    return ws


def basic_template(cfg, template, step, marstype):
    new_template = template.copy()
    new_template.set("bitsPerValue", 24)
    new_template.set("marsType", marstype)
    new_template.set("step", step)
    if step == 0:
        new_template.set("timeRangeIndicator", 1)
    else:
        new_template.set("timeRangeIndicator", 0)
    for key, value in cfg.options["grib_set"].items():
        new_template.set(key, value)
    return new_template


def eps_speed_template(cfg, template, step, number):
    if number == 0:
        eps_template = basic_template(cfg, template, step, "cf")
    else:
        eps_template = basic_template(cfg, template, step, "pf")
        eps_template.set("number", number)
    return eps_template


def write_output(cfg, filename, template, data):
    file = os.path.join(cfg.out_dir, filename)
    target_det = common.target_factory(cfg.target, out_file=file, fdb=cfg.fdb)
    common.write_grib(target_det, template, data)


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.date = datetime.strptime(str(self.options["fc_date"]), "%Y%m%d%H")
        self.root_dir = self.options["root_dir"]
        self.target = self.options["target"]
        self.out_dir = os.path.join(self.root_dir, self.date.strftime("%Y%m%d%H"))

        self.fdb = pyfdb.FDB()

        self.request = self.options["request"]
        self.windows = self.options["windows"]
        self.levelist = self.options.get("levelist", [0])
        self.interpolation_keys = self.options.get("interpolation_keys", None)
        self.vod2uv = bool(self.options.get("vod2uv", False))

        if args.eps_ws or args.eps_mean_std:
            self.members = int(self.options["members"])


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)

    parser = common.default_parser("Calculate wind speed")
    parser.add_argument(
        "--det_ws", action="store_true", default=False, help="Wind speed for type=fc"
    )
    parser.add_argument(
        "--eps_ws", action="store_true", default=False, help="Wind speed for type=pf/cf"
    )
    parser.add_argument(
        "--eps_mean_std",
        action="store_true",
        default=False,
        help="Wind speed mean/std for type=pf/cf. "
        + "Default option if no options are set.",
    )
    args = parser.parse_args(args)

    # If no arguments are selected then run eps_mean_std by default
    if not any([args.det_ws, args.eps_ws, args.eps_mean_std]):
        args.eps_mean_std = True

    cfg = ConfigExtreme(args)

    for levelist in cfg.levelist:
        for window_options in cfg.windows:

            window = common.Window(window_options, include_init=True)

            # calculate wind speed for type=fc (deterministic)
            if args.det_ws:
                for step in window.steps:
                    with common.ResourceMeter(f"Window {window.name}, step {step}, deterministic: read forecast"):
                        messages = fdb_request_det(cfg, levelist, step, window.name)
                    with common.ResourceMeter(f"Window {window.name}, step {step}, deterministic: compute speed"):
                        det = wind_speed(messages)
                    with common.ResourceMeter(f"Window {window.name}, step {step}, deterministic: write output"):
                        template_det = basic_template(cfg, messages[0], step, "fc")
                        write_output(
                            cfg,
                            f"det_{levelist}_{window.name}_{step}.grib",
                            template_det,
                            det[step][0],
                        )

            # calculate wind speed, mean/stddev of wind speed for type=pf/cf (eps)
            if args.eps_ws or args.eps_mean_std:
                for step in window.steps:
                    print(step)
                    with common.ResourceMeter(f"Window {window.name}, step {step}, ensemble: read forecast"):
                        messages = fdb_request_ens(cfg, levelist, step, window.name)
                    with common.ResourceMeter(f"Window {window.name}, step {step}, ensemble: compute speed"):
                        eps = wind_speed(messages)
                    with common.ResourceMeter(f"Window {window.name}, step {step}, ensemble: write output"):
                        template = messages[0]
                        if args.eps_ws:
                            for number in range(cfg.members + 1):
                                template_eps = eps_speed_template(
                                    cfg, template, step, number
                                )
                                write_output(
                                    cfg,
                                    f"eps_{levelist}_{window.name}_{step}_{number}.grib",
                                    template_eps,
                                    eps[step][number],
                                )
                        if args.eps_mean_std:
                            template_mean = basic_template(cfg, template, step, "em")
                            write_output(
                                cfg,
                                f"mean_{levelist}_{window.name}_{step}.grib",
                                template_mean,
                                np.mean(eps[step], axis=0),
                            )

                            template_std = basic_template(cfg, template, step, "es")
                            write_output(
                                cfg,
                                f"std_{levelist}_{window.name}_{step}.grib",
                                template_std,
                                np.std(eps[step], axis=0),
                            )


if __name__ == "__main__":

    main(sys.argv)
