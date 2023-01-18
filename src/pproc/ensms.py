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


def fdb_request_forecast(cfg, param, options, steps):

    req = options['request'].copy()
    req["date"] = cfg.date.strftime("%Y%m%d")
    req["time"] = cfg.date.strftime("%H")+'00'
    req["step"] = steps
    req["param"] = param

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
    pf = common.fdb_read(cfg.fdb, req_pf, mir_options=options.get('interpolation_keys', None))
    print(pf)

    ens = xr.concat([cf, pf], dim='number')
    print(ens)

    return ens


def ensemble_mean_std_eps(cfg, param, options, window):
    """
    Calculate ensemble (type=cf/pf) mean and standard deviation of wind speed
    """
    ens = fdb_request_forecast(cfg, param, options, window.steps)
    template = ens.attrs['grib_template']

    mean = ens.mean(dim='number')
    stddev = ens.std(dim='number')
    print(mean)
    print(stddev)

    return mean, stddev, template


def template_ensemble(cfg, template, marstype):
    template_ens = template.copy()
    template_ens.set("marsType", marstype)
    for key, value in cfg.options['grib_set'].items():
        template_ens.set(key, value)
    return template_ens


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
        for window_options in options['windows']:
            window = common.Window(window_options)

            # calculate mean/stddev of wind speed for type=pf/cf (eps)
            mean, std, template_ens = ensemble_mean_std_eps(cfg, param, options, window)
            for step in window.steps:
                for levelist in options['request']['levelist']:
                    mean_file = os.path.join(cfg.out_dir, f'mean_{param}_{levelist}_{step}.grib')
                    target_mean = common.target_factory(cfg.target, out_file=mean_file, fdb=cfg.fdb)
                    template_mean = template_ensemble(cfg, template_ens, 'em')
                    common.write_grib(target_mean, template_mean, mean.sel(levelist=levelist, step=step).values)
                    std_file = os.path.join(cfg.out_dir, f'std_{param}_{levelist}_{step}.grib')
                    target_std = common.target_factory(cfg.target, out_file=std_file, fdb=cfg.fdb)
                    template_std = template_ensemble(cfg, template_ens, 'es')
                    common.write_grib(target_std, template_std, std.sel(levelist=levelist, step=step).values)


if __name__ == "__main__":
    import sys

    main(sys.argv)
