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
from datetime import datetime
import numpy as np
import signal

import eccodes
from meters import ResourceMeter

from pproc import common
from pproc.common import parallel
from pproc.common.parallel import parallel_processing, sigterm_handler
from pproc.common.utils import dict_product
from pproc.common.param_requester import ParamConfig, ParamRequester


def wind_template(template: eccodes.GRIBMessage, step: int, **out_keys):
    new_template = template.copy()
    grib_sets = {
        "bitsPerValue": 24,
        "step": step,
        **out_keys,
    }
    if step == 0:
        grib_sets["timeRangeIndicator"] = 1
    elif step > 255:
        grib_sets["timeRangeIndicator"] = 10
    else:
        grib_sets["timeRangeIndicator"] = 0

    new_template.set(grib_sets)
    return new_template


class WindParamConfig(ParamConfig):
    def __init__(self, name, options, overrides=None):
        super().__init__(name, options, overrides)
        self.vod2uv = self._in_keys.get("interpolate", {}).get("vod2uv", False)
        self.total_fields = 1
        if self.vod2uv:
            self.in_paramids = [self.in_paramids]
            self.total_fields = 2


class WindConfig(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.members = self.options["num_members"]
        self.total_fields = self.options.get("total_fields", self.members)
        self.date = datetime.strptime(str(self.options["fc_date"]), "%Y%m%d%H")
        self.root_dir = self.options["root_dir"]
        self.sources = self.options.get("sources", {})
        self.det_loc = args.in_det
        self.ens_loc = args.in_ens

        self.n_par = self.options.get("n_par", 1)

        self.out_keys = self.options.get("out_keys", {})
        self.out_keys_em = {"type": "em", **self.options.get("out_keys_em", {})}
        self.out_keys_es = {"type": "es", **self.options.get("out_keys_es", {})}

        self.parameters = [
            WindParamConfig(pname, popt, overrides=self.override_input)
            for pname, popt in self.options["parameters"].items()
        ]
        self.steps = self.options.get("steps", [])
        self.windows = self.options.get("windows", [])

        for attr in ["out_det_ws", "out_eps_ws", "out_eps_mean", "out_eps_std"]:
            location = getattr(args, attr)
            target = common.io.target_from_location(
                location, overrides=self.override_output
            )
            if self.n_par > 1:
                target.enable_parallel(parallel)
            if args.recover:
                target.enable_recovery()
            self.__setattr__(attr, target)


def wind_iteration_gen(
    config: WindConfig,
    loc: str,
    param: WindParamConfig,
    dims: dict,
    members: int,
    total_fields: int,
    out_ws: common.io.Target,
    out_mean=common.io.NullTarget,
    out_std=common.io.NullTarget,
):
    if np.all(
        [isinstance(x, common.io.NullTarget) for x in [out_ws, out_mean, out_std]]
    ):
        return

    requester = ParamRequester(
        param,
        config.sources,
        loc,
        members,
        total_fields * param.total_fields,
    )
    template, ens = requester.retrieve_data(None, **dims)
    assert ens.shape[0] == total_fields, f"Expected {total_fields}, got {ens.shape[0]}"
    with ResourceMeter(f"Param {param.name}, {dims}"):
        if not isinstance(out_ws, common.io.NullTarget):
            for number in range(ens.shape[0]):
                marstype = (
                    "pf"
                    if number > 0 and template.get("type") in ["cf", "fc"]
                    else template.get("type")
                )
                template = wind_template(
                    template,
                    **dims,
                    number=number,
                    type=marstype,
                    **config.out_keys,
                )
                common.write_grib(out_ws, template, ens[number])

        if not isinstance(out_mean, common.io.NullTarget):
            template_mean = wind_template(
                template, **dims, **config.out_keys, **config.out_keys_em
            )
            common.write_grib(out_mean, template_mean, np.mean(ens, axis=0))

        if not isinstance(out_std, common.io.NullTarget):
            template_std = wind_template(
                template, **dims, **config.out_keys, **config.out_keys_es
            )
            common.write_grib(out_std, template_std, np.std(ens, axis=0))


def wind_iteration(
    config: WindConfig, recovery: common.Recovery, param: ParamConfig, dims: dict
):
    # calculate wind speed for type=fc (deterministic)
    wind_iteration_gen(
        config,
        config.det_loc,
        param,
        dims,
        1,
        1,
        config.out_det_ws,
        common.io.NullTarget(),
        common.io.NullTarget(),
    )

    # calculate wind speed, mean/stddev of wind speed for type=pf/cf (eps)
    wind_iteration_gen(
        config,
        config.ens_loc,
        param,
        dims,
        config.members,
        config.total_fields,
        config.out_eps_ws,
        config.out_eps_mean,
        config.out_eps_std,
    )

    for target in [
        config.out_det_ws,
        config.out_eps_ws,
        config.out_eps_mean,
        config.out_eps_std,
    ]:
        target.flush()
    recovery.add_checkpoint(param.name, *dims.values())


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    parser = common.default_parser("Calculate wind speed")
    parser.add_argument(
        "--in-det", required=True, help="Source for deterministic forecast"
    )
    parser.add_argument("--in-ens", required=True, help="Source for ensemble forecast")
    parser.add_argument(
        "--out-det-ws", default="null:", help="Target for wind speed for type=fc"
    )
    parser.add_argument(
        "--out-eps-ws", default="null:", help="Target for wind speed for type=pf/cf"
    )
    parser.add_argument(
        "--out-eps-mean",
        required=True,
        help="Target for mean wind speed for type=pf/cf",
    )
    parser.add_argument(
        "--out-eps-std", required=True, help="Target for wind speed std for type=pf/cf"
    )
    args = parser.parse_args(args)

    cfg = WindConfig(args)
    recovery = common.Recovery(cfg.root_dir, args.config, cfg.date, args.recover)

    plan = []
    for param in cfg.parameters:
        window_manager = common.WindowManager(
            param.window_config(cfg.windows, cfg.steps),
            param.out_keys(cfg.out_keys),
        )
        for dims in dict_product(window_manager.dims):
            if recovery.existing_checkpoint(param.name, *dims.values()):
                print(f"Recovery: skipping dims: {param.name} {dims}")
                continue
            plan.append((param, dims))

    iteration = functools.partial(wind_iteration, cfg, recovery)
    parallel_processing(
        iteration,
        plan,
        cfg.n_par,
        initializer=signal.signal,
        initargs=(signal.SIGTERM, signal.SIG_DFL),
    )

    recovery.clean_file()


if __name__ == "__main__":

    main(sys.argv)
