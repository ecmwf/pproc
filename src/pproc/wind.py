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


def fdb_request_component_ensemble(cfg, levelist, window):

    req = cfg.request.copy()
    req.pop('stream_det')
    req['stream'] = req.pop('stream_ens')

    req['levelist'] = levelist
    req['step'] = window

    req_cf = req.copy()
    req_cf['type'] = 'cf'
    fields_cf = common.fdb_read(cfg.fdb, req_cf)

    req_pf = req.copy()
    req_pf['type'] = 'pf'
    req_pf['number'] = range(1, cfg.members+1)
    fields_pf = common.fdb_read(cfg.fdb, req_pf)

    fields = xr.concat([fields_cf, fields_pf], 'member')

    out = BytesIO()
    inp = mir.MultiDimensionalGribFileInput(target, 2)

    job = mir.Job(vod2uv="1", **pp)
    job.execute(inp, out)

    out.seek(0)
    reader = eccodes.StreamReader(out)
    messages = list(reader)

    return fields


def fdb_request_component_deterministic(cfg, levelist, window, component):

    req = cfg.request.copy()
    req.pop('stream_ens')
    req['stream'] = req.pop('stream_det')

    req['levelist'] = levelist
    req['step'] = window.steps
    req['param'] = req.pop('param')[component]
    req['type'] = 'fc'
    fields = common.fdb_read(cfg.fdb, req)

    return fields


def wind_mean_sd_eps(cfg, levelist, window):
    """
    Calculate ensemble (type=cf/pf) mean and standard deviation of wind speed
    """

    fields_u = fdb_request_component_ensemble(cfg, levelist, window, 0)
    fields_v = fdb_request_component_ensemble(cfg, levelist, window, 1)
    u = fields_u.values
    v = fields_v.values

    ws = np.sqrt(u * u + v * v)
    mean = np.mean(ws, axis=0)
    stddev = np.std(ws, axis=0)

    template = fields_u['grib_template']

    return mean, stddev, template

    # em = messages[0].copy()
    # em.set("marsType", "em")
    # em.set("indicatorOfParameter", "010")  # FIXME check?
    # em.set("gribTablesVersionNo", 128)
    # em.set_array("values", mean)

    # es = messages[0].copy()
    # es.set("marsType", "es")
    # es.set("indicatorOfParameter", "010")  # FIXME check?
    # es.set("gribTablesVersionNo", 128)
    # es.set_array("values", stddev)


def wind_norm_det(cfg, levelist, window):
    """
    Calculate deterministic (type=fc) wind speed
    """

    fields_u = fdb_request_component_deterministic(cfg, levelist, window, 0)
    fields_v = fdb_request_component_deterministic(cfg, levelist, window, 1)
    u = fields_u.values
    v = fields_v.values

    wind_speed = np.sqrt(u * u + v * v)
    template = fields_u['grib_template']

    return wind_speed, template


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.date = datetime.strptime(str(self.options['fc_date']), "%Y%m%d")
        self.root_dir = self.options['root_dir']
        self.out_dir = os.path.join(self.root_dir, 'wind_test', self.fc_date.strftime("%Y%m%d%H"))

        self.fdb = pyfdb.FDB()

        self.request = self.options['request']
        self.interpolation_keys = self.options['interpolation_keys']
        self.windows = self.options['windows']
        self.param = self.options['param']
        self.levelist = self.options['levelist']


def main(args=None):

    parser = common.default_parser('Calculate wind speed mean/standard deviation')
    args = parser.parse_args(args)
    cfg = ConfigExtreme(args)

    for levelist in cfg.levelist:
        for window_options in cfg.windows:

            window = common.Window(window_options)

            # calculate mean/stddev of wind speed for type=pf/cf (eps)
            mean, std, template_ens = wind_mean_sd_eps(cfg, levelist, window)
            mean_file = os.path.join(cfg.out_dir, f'mean_{levelist}_{window}.grib')
            target_mean = common.target_factory(cfg.target, out_file=mean_file, fdb=cfg.fdb)
            template_mean = template_ens
            common.write_grib(target_mean, template_mean, mean)
            std_file = os.path.join(cfg.out_dir, f'std_{levelist}_{window}.grib')
            target_std = common.target_factory(cfg.target, out_file=std_file, fdb=cfg.fdb)
            template_std = template_ens
            common.write_grib(target_std, template_std, std)

            # calculate wind speed for type=fc (deterministic)
            det, template_det = wind_norm_det(cfg, levelist, window)
            det_file = os.path.join(cfg.out_dir, f'det_{levelist}_{window}.grib')
            target_det = common.target_factory(cfg.target, out_file=det_file, fdb=cfg.fdb)
            template_det = template_det
            common.write_grib(target_det, template_det, det)


if __name__ == "__main__":
    import sys

    main(sys.argv)
