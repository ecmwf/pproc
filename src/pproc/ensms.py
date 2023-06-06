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

from pproc import common
from pproc.common.parallel import parallel_processing, sigterm_handler, shared_list


def fdb_request_forecast(cfg, options, steps):

    req = options['request'].copy()
    req["date"] = cfg.date.strftime("%Y%m%d")
    req["time"] = cfg.date.strftime("%H")+'00'
    req["step"] = steps
    req["param"] = options['paramid']

    req_cf = req.copy()
    req_cf['type'] = 'cf'
    print(req_cf)
    cf = common.fdb_read(cfg.fdb, req_cf, mir_options=options.get('interpolation_keys', None))
    print(cf)
    cf = cf.expand_dims(dim={'number': 1})
    cf = cf.assign_coords(number=[0])
    print(cf)

    req_pf = req.copy()
    req_pf['type'] = 'pf'
    req_pf['number'] = range(1, cfg.members+1)
    print(req_pf)
    pf = common.fdb_read(cfg.fdb, req_pf, mir_options=options.get('interpolation_keys', None))
    print(pf)

    ens = xr.concat([cf, pf], dim='number')
    print(ens)

    return ens


def ensemble_mean_std_eps(cfg, options, steps):
    """
    Calculate ensemble (type=cf/pf) mean and standard deviation of wind speed
    """
    ens = fdb_request_forecast(cfg, options, steps)
    template = ens.attrs['grib_template']

    mean = ens.mean(dim='number')
    stddev = ens.std(dim='number')
    print(mean)
    print(stddev)

    return mean, stddev, template


def template_ensemble(cfg, param_type, template, step, window_step, level, marstype):
    template_ens = template.copy()
    template_ens.set('step', step)
    param_type.set_level_key(template_ens, level)
    grib_sets = cfg.options['grib_set']
    if step == 0:
        template_ens.set('timeRangeIndicator', 1)
    elif cfg.options['grib_set'].get('timeRangeIndicator', 0) == 2:
        # Need to set step range 
        template_ens.set({
            'stepType': 'max',
            'stepRange': f'{step - window_step}-{step}',
            'timeRangeIndicator': 2
        })
        grib_sets = cfg.options['grib_set'].copy()
        grib_sets.pop('timeRangeIndicator')
    elif step > 255:
        template_ens.set('timeRangeIndicator', 10)
    else:
        template_ens.set('timeRangeIndicator', 0)
    template_ens.set("marsType", marstype)
    for key, value in grib_sets.items():
        template_ens.set(key, value)
    return template_ens


class PressureLevels:
    def __init__(self, options):
        self.type = 'pl'
        self.levels = options['request']['levelist']

    def set_level_key(self, template, level):
        template.set('level', level)

    def slice_dataset(self, ds, level, **kwargs):
        return ds.sel(levelist=level, **kwargs).values


class SurfaceLevel:
    def __init__(self, options):
        self.levtype = 'sfc'
        self.levels = [0]

    def set_level_key(self, template, level):
        return

    def slice_dataset(self, ds, level, **kwargs):
        return ds.sel(**kwargs).values


def parameters_manager(options):
    if options['request']['levtype'] == 'pl':
        param_type = PressureLevels(options)
    else:
        param_type = SurfaceLevel(options)
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
            if self.n_par > 1 and type(target) in [common.io.FileTarget, common.io.FileSetTarget]:
                target.track_truncated = shared_list()
            self.__setattr__(attr, target)

    @property
    def fdb(self):
        if self._fdb is None:
            self._fdb = common.io.fdb()
        return self._fdb


def ensms_iteration(config, param, options, window, step):
    param_type = parameters_manager(options)

    # calculate mean/stddev of wind speed for type=pf/cf (eps)
    with common.ResourceMeter(f"Window {window.name}, step {step}: compute mean/stddev"):
        mean, std, template_ens = ensemble_mean_std_eps(config, options, step)

    with common.ResourceMeter(f"Window {window.name}, step {step}: write output"):
        for level in param_type.levels:
            mean_slice = param_type.slice_dataset(mean, level)
            template_mean = template_ensemble(config, param_type, template_ens, step, window.step, level, 'em')
            common.write_grib(config.out_eps_mean, template_mean, mean_slice)

            std_slice = param_type.slice_dataset(std, level)
            template_std = template_ensemble(config, param_type, template_ens, step, window.step, level, 'es')
            common.write_grib(config.out_eps_std, template_std, std_slice)

    config.fdb.flush()
    return param, window.name, step


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

    plan = []
    for param, options in cfg.parameters.items():
        for window_options in options['windows']:
            window = common.Window(window_options)

            for step in window.steps:
                if recover.existing_checkpoint(param, window.name, step):
                    print(f'Recovery: skipping param {param} step {step}')
                    continue

                plan.append((param, options, window, step))

    iteration = functools.partial(ensms_iteration, cfg)
    parallel_processing(iteration, plan, cfg.n_par, recover)

    recover.clean_file()

if __name__ == "__main__":
    main(sys.argv)
