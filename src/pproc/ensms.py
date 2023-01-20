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
from datetime import datetime
import numpy as np
import xarray as xr

import pyfdb

from pproc import common


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


def ensemble_mean_std_eps(cfg, options, window):
    """
    Calculate ensemble (type=cf/pf) mean and standard deviation of wind speed
    """
    ens = fdb_request_forecast(cfg, options, window.steps)
    template = ens.attrs['grib_template']

    mean = ens.mean(dim='number')
    stddev = ens.std(dim='number')
    print(mean)
    print(stddev)

    return mean, stddev, template


def template_ensemble(cfg, param_type, template, step, level, marstype):
    template_ens = template.copy()
    template_ens.set('step', step)
    param_type.set_level_key(template_ens, level)
    if step == 0:
        template_ens.set('timeRangeIndicator', 1)
    else:
        template_ens.set('timeRangeIndicator', 0)
    template_ens.set("marsType", marstype)
    for key, value in cfg.options['grib_set'].items():
        template_ens.set(key, value)
    return template_ens


class PressureLevels:
    def __init__(self, options):
        self.type = 'pl'
        self.levels = options['request']['levelist']

    def set_level_key(self, template, level):
        template.set('level', level)

    def slice_dataset(self, ds, level, step):
        return ds.sel(levelist=level, step=step).values


class SurfaceLevel:
    def __init__(self, options):
        self.levtype = 'sfc'
        self.levels = [0]

    def set_level_key(self, template, level):
        return

    def slice_dataset(self, ds, level, step):
        return ds.sel(step=step).values


def parameters_manager(options):
    if options['request']['levtype'] == 'pl':
        param_type = PressureLevels(options)
    else:
        param_type = SurfaceLevel(options)
    return param_type


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.members = self.options['members']
        self.date = datetime.strptime(str(self.options['fc_date']), "%Y%m%d%H")
        self.root_dir = self.options['root_dir']
        self.target = self.options['target']
        self.out_dir = os.path.join(self.root_dir, self.date.strftime("%Y%m%d%H"))

        self.fdb = pyfdb.FDB()

        self.parameters = self.options['parameters']


def main(args=None):

    parser = common.default_parser('Calculate wind speed mean/standard deviation')
    args = parser.parse_args(args)
    cfg = ConfigExtreme(args)

    for param, options in cfg.parameters.items():
        
        param_type = parameters_manager(options)

        for window_options in options['windows']:
            window = common.Window(window_options)

            # calculate mean/stddev of wind speed for type=pf/cf (eps)
            mean, std, template_ens = ensemble_mean_std_eps(cfg, options, window)

            for step in window.steps:
                for level in param_type.levels:
                    mean_slice = param_type.slice_dataset(mean, level, step)
                    mean_file = os.path.join(cfg.out_dir, f'mean_{param}_{level}_{step}.grib')
                    target_mean = common.target_factory(cfg.target, out_file=mean_file, fdb=cfg.fdb)
                    template_mean = template_ensemble(cfg, param_type, template_ens, step, level, 'em')
                    common.write_grib(target_mean, template_mean, mean_slice)

                    std_slice = param_type.slice_dataset(std, level, step)
                    std_file = os.path.join(cfg.out_dir, f'std_{param}_{level}_{step}.grib')
                    target_std = common.target_factory(cfg.target, out_file=std_file, fdb=cfg.fdb)
                    template_std = template_ensemble(cfg, param_type, template_ens, step, level, 'es')
                    common.write_grib(target_std, template_std, std_slice)


if __name__ == "__main__":
    import sys

    main(sys.argv)
