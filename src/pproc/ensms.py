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
import xarray as xr
import signal
import numpy as np

from pproc import common
from pproc.common.parallel import (
    parallel_processing, 
    sigterm_handler, 
    shared_list,
    SynchronousExecutor,
    QueueingExecutor,
    parallel_data_retrieval
)


class ParamRequester:
    def __init__(self, param, cfg, options):
        self.name = param
        self.date = cfg.date
        self.members = cfg.members
        self.options = options

    def retrieve_data(self, fdb, step):
        req = self.options['request'].copy()
        req["date"] = self.date.strftime("%Y%m%d")
        req["time"] = self.date.strftime("%H")+'00'
        req["step"] = step
        req["param"] = self.options['paramid']

        req_cf = req.copy()
        req_cf['type'] = 'cf'
        print(req_cf)
        cf = common.fdb_read(fdb, req_cf, mir_options=self.options.get('interpolation_keys', None))
        cf = cf.expand_dims(dim={'number': 1})
        cf = cf.assign_coords(number=[0])

        req_pf = req.copy()
        req_pf['type'] = 'pf'
        req_pf['number'] = range(1, self.members+1)
        print(req_pf)
        pf = common.fdb_read(fdb, req_pf, mir_options=self.options.get('interpolation_keys', None))

        ens = xr.concat([cf, pf], dim='number')
        template = ens.attrs['grib_template']
        del ens.attrs['grib_template']
        return template, ens


def template_ensemble(cfg, param_type, template, window, level, marstype):
    template_ens = template.copy()
    param_type.set_level_key(template_ens, level)

    if window.size() == 0:
        step = int(window.name)
        template_ens.set('step', step)
        if step == 0:
            template_ens.set('timeRangeIndicator', 1)
        elif step > 255:
            template_ens.set('timeRangeIndicator', 10)
        else:
            template_ens.set('timeRangeIndicator', 0)
    else:
        template_ens.set({
            'stepType': 'max',
            'stepRange': window.name,
            'timeRangeIndicator': 2
        })
       
    template_ens.set("marsType", marstype)
    for key, value in cfg.options['grib_set'].items():
        template_ens.set(key, value)
    return template_ens


class PressureLevels(ParamRequester):
    def __init__(self, param, cfg, options):
        super().__init__(param, cfg, options)
        self.type = 'pl'
        self.levels = options['request']['levelist']

    def set_level_key(self, template, level):
        template.set('level', level)

    def slice_dataset(self, ds, level, **kwargs):
        return ds.sel(levelist=level, **kwargs).values


class SurfaceLevel(ParamRequester):
    def __init__(self, param, cfg, options):
        super().__init__(param, cfg, options)
        self.levtype = 'sfc'
        self.levels = [0]

    def set_level_key(self, template, level):
        return

    def slice_dataset(self, ds, level, **kwargs):
        return ds.sel(**kwargs).values


def parameters_manager(param, config, options):
    if options['request']['levtype'] == 'pl':
        param_type = PressureLevels(param, config, options)
    else:
        param_type = SurfaceLevel(param, config, options)
    return param_type


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


def ensms_iteration(config, param_type, recovery, window_id, window, template_ens = None):
    # calculate mean/stddev of wind speed for type=pf/cf (eps)
    with common.ResourceMeter(f"Window {window.name}: compute mean/stddev"):
        if template_ens is None:
            template_ens, ens = param_type.retrieve_data(config.fdb, window.steps[0])
        else:
            if isinstance(template_ens, str):
                template_ens = common.io.read_template(template_ens)
            ens = window.step_values
        mean = ens.mean(dim='number')
        std = ens.std(dim='number')

    with common.ResourceMeter(f"Window {window.name}: write output"):
        for level in param_type.levels:
            mean_slice = param_type.slice_dataset(mean, level)
            template_mean = template_ensemble(config, param_type, template_ens, window, level, 'em')
            common.write_grib(config.out_eps_mean, template_mean, mean_slice)

            std_slice = param_type.slice_dataset(std, level)
            template_std = template_ensemble(config, param_type, template_ens, window, level, 'es')
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
        param_type = parameters_manager(param, cfg, options)
        window_manager = common.WindowManager(options, {})
        iteration = functools.partial(ensms_iteration, cfg, param_type, recover)

        if np.all([len(x.steps) == 1 for x in window_manager.windows.values()]): 
            plan = []
            for window_id, window in window_manager.windows.items():
                if recover.existing_checkpoint(param, window.name):
                    print(f'Recovery: skipping param {param} window {window.name}')
                    continue

                plan.append((window_id, window))

            parallel_processing(iteration, plan, cfg.n_par)
        else:
            executor = (
                SynchronousExecutor()
                if cfg.n_par == 1
                else QueueingExecutor(cfg.n_par, cfg.n_par)
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
                    window_manager.delete_windows(checkpointed_windows)
                    print(
                        f"Recovery: param {param} looping from step {window_manager.unique_steps[0]}"
                    )
                    last_checkpoint = None  # All remaining params have not been run

                for step, retrieved_data in parallel_data_retrieval(
                    cfg.n_par,
                    window_manager.unique_steps,
                    [param_type],
                    cfg.n_par > 1,
                ):
                    with common.ResourceMeter(f"Process step {step}"):
                        message_template, data = retrieved_data[0]

                        completed_windows = window_manager.update_windows(
                            step,
                            data,
                        )
                        for window_id, window in completed_windows:
                            executor.submit(
                                iteration,
                                window_id,
                                window, 
                                message_template
                            )
                executor.wait()

    recover.clean_file()

if __name__ == "__main__":
    main(sys.argv)
