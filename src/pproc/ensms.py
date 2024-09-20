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
import signal
from meters import ResourceMeter
import numpy as np
from typing import Union

import eccodes

from pproc import common
from pproc.common import parallel
from pproc.common.accumulation import Accumulator
from pproc.common.parallel import (
    parallel_processing,
    sigterm_handler,
    SynchronousExecutor,
    QueueingExecutor,
    parallel_data_retrieval,
)
from pproc.common.param_requester import ParamConfig, ParamRequester


def template_ensemble(
    param_type: ParamConfig,
    template: eccodes.GRIBMessage,
    accum: Accumulator,
    marstype: str,
):
    template_ens = template.copy()

    grib_sets = accum.grib_keys()
    grib_sets["marsType"] = marstype
    if template_ens["edition"] == 2 or grib_sets.get("edition", 1) == 2:
        grib_sets["productDefinitionTemplateNumber"] = 2
        if marstype in ["em", "es"]:
            grib_sets["derivedForecast"] = 2 if marstype == "es" else 0
    template_ens.set(grib_sets)
    return template_ens


class EnsmsConfig(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.members = int(self.options["num_members"])
        self.total_fields = self.options.get("total_fields", self.members)
        self.date = datetime.strptime(str(self.options["fc_date"]), "%Y%m%d%H")
        self.root_dir = self.options["root_dir"]
        self.sources = self.options.get("sources", {})

        self.n_par_read = self.options.get("n_par_read", 1)
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)

        self._fdb = None

        self.out_keys = self.options.get("out_keys", {})

        self.parameters = [
            ParamConfig(pname, popt, overrides=self.override_input)
            for pname, popt in self.options["params"].items()
        ]
        self.steps = self.options.get("steps", [])
        self.windows = self.options.get("windows", [])

        for attr in ["out_mean", "out_std"]:
            location = getattr(args, attr)
            target = common.io.target_from_location(
                location, overrides=self.override_output
            )
            if self.n_par_compute > 1:
                target.enable_parallel(parallel)
            if args.recover:
                target.enable_recovery()
            self.__setattr__(attr, target)

    @property
    def fdb(self):
        if self._fdb is None:
            self._fdb = common.io.fdb()
        return self._fdb


def ensms_iteration(
    config: EnsmsConfig,
    param: ParamConfig,
    recovery: common.Recovery,
    window_id: str,
    accum: Accumulator,
    template=Union[str, eccodes.GRIBMessage],
):
    if not isinstance(template, eccodes.GRIBMessage):
        template_ens = common.io.read_template(template)

    ens = accum.values
    assert ens is not None

    # Compute mean/std over all dimensions except last
    axes = tuple(range(ens.ndim - 1))
    with ResourceMeter(f"Window {window_id}: write mean output"):
        mean = np.mean(ens, axis=axes)
        template_mean = template_ensemble(param, template_ens, accum, "em")
        template_mean.set_array("values", common.io.nan_to_missing(template_mean, mean))
        config.out_mean.write(template_mean)
        config.out_mean.flush()

    with ResourceMeter(f"Window {window_id}: write std output"):
        std = np.std(ens, axis=axes)
        template_std = template_ensemble(param, template_ens, accum, "es")
        template_std.set_array("values", common.io.nan_to_missing(template_std, std))
        config.out_std.write(template_std)
        config.out_std.flush()

    recovery.add_checkpoint(param.name, window_id)


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    parser = common.default_parser("Calculate mean/standard deviation")
    parser.add_argument("--in-ens", required=True, help="Input ensemble")
    parser.add_argument("--out-mean", required=True, help="Target for mean")
    parser.add_argument(
        "--out-std", required=True, help="Target for standard deviation"
    )
    args = parser.parse_args(args)
    cfg = EnsmsConfig(args)
    recover = common.Recovery(cfg.root_dir, args.config, cfg.date, args.recover)
    last_checkpoint = recover.last_checkpoint()

    executor = (
        SynchronousExecutor()
        if cfg.n_par_compute == 1
        else QueueingExecutor(
            cfg.n_par_compute,
            cfg.window_queue_size,
            initializer=signal.signal,
            initargs=(signal.SIGTERM, signal.SIG_DFL),
        )
    )

    with executor:
        for param in cfg.parameters:
            out_key_kwargs = {"paramId": param.out_paramid} if param.out_paramid else {}
            window_manager = common.WindowManager(
                param.window_config(cfg.windows, cfg.steps),
                param.out_keys(cfg.out_keys, **out_key_kwargs),
            )

            if last_checkpoint:
                if param.name not in last_checkpoint:
                    print(f"Recovery: skipping completed param {param.name}")
                    continue
                checkpointed_windows = [
                    recover.checkpoint_identifiers(x)[1]
                    for x in recover.checkpoints
                    if param.name in x
                ]
                new_start = window_manager.delete_windows(checkpointed_windows)
                print(f"Recovery: param {param.name} looping from step {new_start}")
                last_checkpoint = None  # All remaining params have not been run

            requester = ParamRequester(
                param,
                cfg.sources,
                args.in_ens,
                cfg.members,
                cfg.total_fields,
            )
            iteration = functools.partial(ensms_iteration, cfg, param, recover)
            for keys, retrieved_data in parallel_data_retrieval(
                cfg.n_par_read,
                window_manager.dims,
                [requester],
                cfg.n_par_compute > 1,
                initializer=signal.signal,
                initargs=(signal.SIGTERM, signal.SIG_DFL),
            ):
                step = keys["step"]
                with ResourceMeter(f"Process step {step}"):
                    message_template, data = retrieved_data[0]

                    completed_windows = window_manager.update_windows(
                        keys,
                        data,
                    )
                    for window_id, accum in completed_windows:
                        executor.submit(iteration, window_id, accum, message_template)
            executor.wait()

    recover.clean_file()


if __name__ == "__main__":
    main(sys.argv)
