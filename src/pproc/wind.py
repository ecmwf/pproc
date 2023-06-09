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
import functools
import sys
from io import BytesIO
from datetime import datetime
import numpy as np
import signal

import eccodes
import mir

from pproc import common
from pproc.common.parallel import parallel_processing, sigterm_handler, shared_list


def retrieve_messages(cfg, req, cached_file):
    if cfg.vod2uv:
        print(req)
        common.fdb_read_to_file(cfg.fdb, req, cached_file)
        messages = mir_wind(cfg, cached_file)
    else:
        out = common.fdb_retrieve(cfg.fdb, req, cfg.interpolation_keys)
        reader = eccodes.StreamReader(out)
        if not reader.peek():
            raise RuntimeError(f"No data retrieved for request {req}")
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

    try:
        stepid = list(steps)[0]
    except (TypeError, ValueError):
        stepid = steps

    return retrieve_messages(cfg, req, f"wind_det_{levelist}_{name}_{stepid}.grb")


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

    try:
        stepid = list(steps)[0]
    except (TypeError, ValueError):
        stepid = steps

    req_cf = req.copy()
    req_cf["type"] = "cf"
    messages = retrieve_messages(cfg, req_cf, f"wind_cf_{levelist}_{name}_{stepid}.grb")

    req_pf = req.copy()
    req_pf["type"] = "pf"
    req_pf["number"] = range(1, cfg.members + 1)
    messages += retrieve_messages(cfg, req_pf, f"wind_pf_{levelist}_{name}_{stepid}.grb")

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
    elif step > 255:
        new_template.set('timeRangeIndicator', 10)
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


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.date = datetime.strptime(str(self.options["fc_date"]), "%Y%m%d%H")
        self.root_dir = self.options["root_dir"]

        self.n_par = self.options.get("n_par", 1)
        self._fdb = None

        self.request = self.options["request"]
        self.windows = self.options["windows"]
        self.levelist = self.options.get("levelist", [0])
        self.interpolation_keys = self.options.get("interpolation_keys", None)
        self.vod2uv = bool(self.options.get("vod2uv", False))

        self.members = int(self.options["members"]) if "members" in self.options else None

        for attr in ["out_det_ws", "out_eps_ws", "out_eps_mean", "out_eps_std"]:
            location = getattr(args, attr)
            target = common.io.target_from_location(location)
            if type(target) in [common.io.FileTarget, common.io.FileSetTarget]:
                if self.n_par > 1:
                    target.track_truncated = shared_list()
                if args.recover:
                    target.enable_recovery()
            self.__setattr__(attr, target)

    @property
    def fdb(self):
        if self._fdb is None:
            self._fdb = common.io.fdb()
        return self._fdb


def wind_iteration_gen(config, tp, levelist, name, step, out_ws, out_mean=common.io.NullTarget, out_std=common.io.NullTarget):
    if np.all([isinstance(x, common.io.NullTarget) for x in [out_ws, out_mean, out_std]]):
        return

    fdb_req = fdb_request_det if tp == "det" else fdb_request_ens
    tpname = "deterministic" if tp == "det" else "ensemble"
    mk_template = (
        (lambda cfg, msg, stp, num: basic_template(cfg, msg, stp, "fc"))
        if tp == "det"
        else eps_speed_template
    )
    numbers = [0] if tp == "det" else range(config.members + 1)

    with common.ResourceMeter(
        f"Window {name}, step {step}, {tpname}: read forecast"
    ):
        messages = fdb_req(config, levelist, step, name)
    with common.ResourceMeter(
        f"Window {name}, step {step}, {tpname}: compute speed"
    ):
        spd = wind_speed(messages)
    with common.ResourceMeter(
        f"Window {name}, step {step}, {tpname}: write output"
    ):
        template = messages[0]
        if not isinstance(out_ws, common.io.NullTarget):
            for number in numbers:
                template = mk_template(config, template, step, number)
                common.write_grib(out_ws, template, spd[step][number])

        if not isinstance(out_mean, common.io.NullTarget):
            template_mean = basic_template(config, template, step, "em")
            common.write_grib(out_mean, template_mean, np.mean(spd[step], axis=0))

        if not isinstance(out_std, common.io.NullTarget):
            template_std = basic_template(config, template, step, "es")
            common.write_grib(out_std, template_std, np.std(spd[step], axis=0))


def wind_iteration(config, recovery, levelist, name, step):
    # calculate wind speed for type=fc (deterministic)
    wind_iteration_gen(config, "det", levelist, name, step, config.out_det_ws, common.io.NullTarget(), common.io.NullTarget())

    # calculate wind speed, mean/stddev of wind speed for type=pf/cf (eps)
    wind_iteration_gen(config, "eps", levelist, name, step, config.out_eps_ws, config.out_eps_mean, config.out_eps_std)

    config.fdb.flush()
    recovery.add_checkpoint(levelist, name, step)


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    parser = common.default_parser("Calculate wind speed")
    parser.add_argument(
        "--out_det_ws", default="null:", help="Target for wind speed for type=fc"
    )
    parser.add_argument(
        "--out_eps_ws", default="null:", help="Target for wind speed for type=pf/cf"
    )
    parser.add_argument(
        "--out_eps_mean", required=True, help="Target for mean wind speed for type=pf/cf"
    )
    parser.add_argument(
        "--out_eps_std", required=True, help="Target for wind speed std for type=pf/cf"
    )
    args = parser.parse_args(args)

    cfg = ConfigExtreme(args)
    recovery = common.Recovery(cfg.root_dir, args.config, cfg.date, args.recover)

    plan = []
    for levelist in cfg.levelist:
        for window_options in cfg.windows:

            window = common.Window(window_options, include_init=True)

            for step in window.steps:
                if recovery.existing_checkpoint(levelist, window.name, step):
                    print(
                        f"Recovery: skipping level {levelist} window {window} step {step}"
                    )
                    continue

                plan.append((levelist, window.name, step))

    iteration = functools.partial(wind_iteration, cfg, recovery)
    parallel_processing(iteration, plan, cfg.n_par)

    recovery.clean_file()


if __name__ == "__main__":

    main(sys.argv)
