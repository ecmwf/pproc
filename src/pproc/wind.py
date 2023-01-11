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
from io import BytesIO
from datetime import datetime
import numpy as np
import xarray as xr

import eccodes
import pyfdb
import mir

from pproc import common


def fdb2cache_request_det(cfg, levelist, window):

    req = cfg.request.copy()
    req.pop('stream_ens')
    req['stream'] = req.pop('stream_det')

    req['levelist'] = levelist
    req['step'] = window.steps
    req['type'] = 'fc'

    cached_file = f"{req['param']}_det.grb"
    common.fdb_read_to_file(cfg.fdb, req, cached_file)

    return cached_file


def fdb2cache_request_ens(cfg, levelist, window):

    req = cfg.request.copy()
    req.pop('stream_det')
    req['stream'] = req.pop('stream_ens')

    req['levelist'] = levelist
    req['step'] = window.steps

    req_cf = req.copy()
    req_cf['type'] = 'cf'

    cached_file = f"wind_ens_{levelist}_{window.name}.grb"
    common.fdb_read_to_file(cfg.fdb, req, cached_file)

    req_pf = req.copy()
    req_pf['type'] = 'pf'
    req_pf['number'] = range(1, cfg.members+1)
    common.fdb_read_to_file(cfg.fdb, req, cached_file, mode='ab')

    return cached_file


def mir_wind(cached_file):

    
    out = BytesIO()
    inp = mir.MultiDimensionalGribFileInput(cached_file, 2)

    job = mir.Job(vod2uv="1")
    job.execute(inp, out)

    out.seek(0)
    reader = eccodes.StreamReader(out)
    messages = list(reader)

    wind_paramids = set([m["paramId"] for m in messages])
    assert len(wind_paramids) == 2
    u = np.asarray([m.get_array("values") for m in messages if m["paramId"] == wind_paramids[0]])
    v = np.asarray([m.get_array("values") for m in messages if m["paramId"] == wind_paramids[1]])

    template = messages[0]

    return u, v, template


def wind_norm_det(cfg, levelist, window):
    """
    Calculate deterministic (type=fc) wind speed
    """
    cached_file = fdb2cache_request_det(cfg, levelist, window)

    u, v, template = mir_wind(cached_file)

    wind_speed = np.sqrt(u * u + v * v)

    return wind_speed, template


def wind_mean_std_eps(cfg, levelist, window):
    """
    Calculate ensemble (type=cf/pf) mean and standard deviation of wind speed
    """
    cached_file = fdb2cache_request_ens(cfg, levelist, window)

    u, v, template = mir_wind(cached_file)

    ws = np.sqrt(u * u + v * v)
    mean = np.mean(ws, axis=0)
    stddev = np.std(ws, axis=0)

    return mean, stddev, template


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
            mean, std, template_ens = wind_mean_std_eps(cfg, levelist, window)
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
