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


def template_ensemble(param_type, template, accum, level, marstype):
    template_ens = template.copy()

    grib_sets = accum.grib_keys()
    if param_type.base_request['levtype'] == "pl":
        grib_sets['level'] = level

    grib_sets["marsType"] = marstype
    template_ens.set(grib_sets)
    return template_ens

def slice_dataset(ds, level_index):
    if ds.ndim > 1:
        return ds[level_index]
    return ds

class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.members = int(self.options['members'])
        self.date = datetime.strptime(str(self.options['fc_date']), "%Y%m%d%H")
        self.root_dir = self.options['root_dir']

        self.n_par = self.options.get("n_par", 1)
        self._fdb = None

        self.parameters = self.options['parameters']

        for attr in ["out_eps_mean", "out_eps_std"]:
            location = getattr(args, attr)
            target = common.io.target_from_location(location, overrides=self.override_output)
            if self.n_par > 1:
                target.enable_parallel(parallel)
            if args.recover:
                target.enable_recovery()
            self.__setattr__(attr, target)

    @property
    def fdb(self):
        if self._fdb is None:
            self._fdb = common.io.fdb()
        return self._fdb


def ensms_iteration(config, param_type, recovery, window_id, accum, template_ens = None):
    # calculate mean/stddev of wind speed for type=pf/cf (eps)
    with ResourceMeter(f"Window {window_id}: compute mean/stddev"):
        if template_ens is None:
            template_ens, ens = param_type.retrieve_data(config.fdb, accum.dims[0].accumulation.coords[0])
        else:
            if isinstance(template_ens, str):
                template_ens = common.io.read_template(template_ens)
            ens = accum.values
            assert ens is not None
        axes = tuple(range(ens.ndim - 1))
        mean = np.mean(ens, axis=axes)
        std = np.std(ens, axis=axes)

    with ResourceMeter(f"Window {window_id}: write output"):
        for level_index, level in enumerate(param_type.levels()):
            mean_slice = slice_dataset(mean, level_index)
            template_mean = template_ensemble(param_type, template_ens, accum, level, 'em')
            common.write_grib(config.out_eps_mean, template_mean, mean_slice)

            std_slice = slice_dataset(std, level_index)
            template_std = template_ensemble(param_type, template_ens, accum, level, 'es')
            common.write_grib(config.out_eps_std, template_std, std_slice)

    config.fdb.flush()
    recovery.add_checkpoint(param_type.name, window_id)


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    parser = common.default_parser('Calculate mean/standard deviation')
    parser.add_argument(
        "--out_eps_mean", required=True, help="Target for mean"
    )
    parser.add_argument(
        "--out_eps_std", required=True, help="Target for standard deviation"
    )
    args = parser.parse_args(args)
    cfg = ConfigExtreme(args)
    recover = common.Recovery(cfg.root_dir, args.config, cfg.date, args.recover)
    last_checkpoint = recover.last_checkpoint()

    for param, options in cfg.parameters.items():
        param_type = common.parameter.create_parameter(
            param, cfg.date, {}, options, cfg.members, cfg.override_input
        )
        window_manager = common.WindowManager(options, cfg.options["grib_set"])
        iteration = functools.partial(ensms_iteration, cfg, param_type, recover)

        if np.all([len(x) == 1 for x in window_manager.mgr.accumulations.values()]):
            plan = []
            for window_id, accum in window_manager.mgr.accumulations.items():
                if recover.existing_checkpoint(param, window_id):
                    print(f'Recovery: skipping param {param} window {window_id}')
                    continue

                plan.append((window_id, accum))

            parallel_processing(iteration, plan, cfg.n_par, initializer=signal.signal,
                                initargs=(signal.SIGTERM, signal.SIG_DFL))
        else:
            executor = (
                SynchronousExecutor()
                if cfg.n_par == 1
                else QueueingExecutor(cfg.n_par, cfg.n_par, initializer=signal.signal,
                                      initargs=(signal.SIGTERM, signal.SIG_DFL))
            )

            with executor:
                if last_checkpoint:
                    if param not in last_checkpoint:
                        print(f"Recovery: skipping completed param {param}")
                        continue
                    checkpointed_windows = [
                        recover.checkpoint_identifiers(x)[1]
                        for x in recover.checkpoints
                        if param in x
                    ]
                    new_start = window_manager.delete_windows(checkpointed_windows)
                    print(
                        f"Recovery: param {param} looping from step {new_start}"
                    )
                    last_checkpoint = None  # All remaining params have not been run

                for keys, retrieved_data in parallel_data_retrieval(
                    cfg.n_par,
                    window_manager.dims,
                    [param_type],
                    cfg.n_par > 1, 
                    initializer=signal.signal,
                    initargs=(signal.SIGTERM, signal.SIG_DFL)
                ):
                    step = keys["step"]
                    with ResourceMeter(f"Process step {step}"):
                        message_template, data = retrieved_data[0]

                        completed_windows = window_manager.update_windows(
                            keys,
                            data,
                        )
                        for window_id, accum in completed_windows:
                            executor.submit(
                                iteration,
                                window_id,
                                accum,
                                message_template
                            )
                executor.wait()

    recover.clean_file()

if __name__ == "__main__":
    main(sys.argv)
