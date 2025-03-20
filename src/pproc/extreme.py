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


import sys
from datetime import datetime
import functools
import signal
from typing import Dict, Any, Union

import eccodes
from meters import ResourceMeter
from pproc import common
from pproc.common import parallel
from pproc.common.accumulation import Accumulator
from pproc.common.parallel import (
    SynchronousExecutor,
    QueueingExecutor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.param_requester import ParamConfig, ParamRequester
from pproc.extremes.grib import extreme_template
from pproc.extremes.indices import SUPPORTED_INDICES, create_indices
from pproc.signi.clim import retrieve_clim


DEFAULT_INDICES = ["efi", "sot"]


class ExtremeParamConfig(ParamConfig):
    def __init__(
        self, name: str, options: Dict[str, Any], overrides: Dict[str, Any] = {}
    ):
        options = options.copy()
        clim_options = options.pop("clim")
        super().__init__(name, options, overrides)
        self.clim_param = ParamConfig(f"clim_{name}", clim_options)
        self.indices = create_indices(options, DEFAULT_INDICES)


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.fc_date = datetime.strptime(str(self.options["fc_date"]), "%Y%m%d%H")
        self.members = self.options.get("num_members", 51)
        self.total_fields = self.options.get("total_fields", self.members)
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.n_par_read = self.options.get("n_par_read", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)

        self.root_dir = self.options["root_dir"]
        self.out_keys = self.options.get("out_keys", {})

        self.sources = self.options.get("sources", {})

        self.steps = self.options.get("steps", [])
        self.windows = self.options.get("windows", [])

        self.parameters = [
            ExtremeParamConfig(pname, popt, overrides=self.override_input)
            for pname, popt in self.options["parameters"].items()
        ]
        self.clim_loc = args.in_clim

        self.targets = {}
        for index in SUPPORTED_INDICES:
            attr = f"out_{index}"
            location = getattr(args, attr)
            target = common.io.target_from_location(
                location, overrides=self.override_output
            )
            if self.n_par_compute > 1:
                target.enable_parallel(parallel)
            if args.recover:
                target.enable_recovery()
            self.targets[index] = target


def read_clim(
    config: ConfigExtreme,
    param: ExtremeParamConfig,
    accum: Accumulator,
    n_clim: int = 101,
):
    grib_keys = accum.grib_keys()
    clim_step = grib_keys.get("stepRange", grib_keys.get("step", None))
    in_keys = param.clim_param._in_keys
    in_keys["quantile"] = ["{}:100".format(i) for i in range(n_clim)]
    step = in_keys.get("step", {}).get(clim_step, clim_step)
    clim_accum, clim_template = retrieve_clim(
        param.clim_param,
        config.sources,
        config.clim_loc,
        1,
        n_clim,
        index_func=lambda x: int(x.get("quantile").split(":")[0]),
        step=step,
    )
    if not isinstance(clim_template, eccodes.GRIBMessage):
        clim_template = common.io.read_template(clim_template)
    return clim_accum.values, clim_template


def compute_indices(
    cfg: ConfigExtreme,
    param: ExtremeParamConfig,
    recovery: common.Recovery,
    template_filename: Union[str, eccodes.GRIBMessage],
    window_id: str,
    accum: Accumulator,
):
    with ResourceMeter(f"Window {window_id}, computing indices"):
        message_template = (
            template_filename
            if isinstance(template_filename, eccodes.GRIBMessage)
            else common.io.read_template(template_filename)
        )

        clim, template_clim = read_clim(cfg, param, accum)
        print(f"Climatology array: {clim.shape}")

        template_extreme = extreme_template(accum, message_template, template_clim)

        ens = accum.values
        assert ens is not None

        for name, index in param.indices.items():
            target = cfg.targets[name]
            index.compute(clim, ens, target, message_template, template_extreme)
            target.flush()

        recovery.add_checkpoint(param.name, window_id)


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    parser = common.default_parser(
        "Compute extreme indices from forecast and climatology"
    )
    parser.add_argument("--in-ens", required=True, help="Source for forecast")
    parser.add_argument("--in-clim", required=True, help="Source for climatology")
    for index in SUPPORTED_INDICES:
        parser.add_argument(
            f"--out-{index}", default="null:", help=f"Target for {index.upper()}"
        )
    args = parser.parse_args(args)
    cfg = ConfigExtreme(args)
    recovery = common.Recovery(cfg.root_dir, args.config, cfg.fc_date, args.recover)
    last_checkpoint = recovery.last_checkpoint()
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
            requester = ParamRequester(
                param, cfg.sources, args.in_ens, cfg.members, cfg.total_fields
            )
            window_manager = common.WindowManager(
                param.window_config(cfg.windows, cfg.steps),
                param.out_keys(cfg.out_keys),
            )
            if last_checkpoint:
                if param.name not in last_checkpoint:
                    print(f"Recovery: skipping completed param {param.name}")
                    continue
                checkpointed_windows = [
                    recovery.checkpoint_identifiers(x)[1]
                    for x in recovery.checkpoints
                    if param.name in x
                ]
                new_start = window_manager.delete_windows(checkpointed_windows)
                print(f"Recovery: param {param.name} looping from step {new_start}")
                last_checkpoint = None  # All remaining params have not been run

            indices_partial = functools.partial(compute_indices, cfg, param, recovery)
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
                    template, data = retrieved_data[0]
                    assert data.ndim == 2

                    completed_windows = window_manager.update_windows(keys, data)
                    for window_id, accum in completed_windows:
                        executor.submit(indices_partial, template, window_id, accum)

            executor.wait()

        recovery.clean_file()


if __name__ == "__main__":
    main()
