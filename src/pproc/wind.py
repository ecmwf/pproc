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
    """
    Retrieve vorticity and divergence from FDB and dump it to disk
    Deterministic forecast
    """

    req = cfg.request.copy()
    req.pop('stream_ens')
    req['stream'] = req.pop('stream_det')
    req["date"] = cfg.date.strftime("%Y%m%d")
    req["time"] = cfg.date.strftime("%H")+'00'
    req['levelist'] = levelist
    req['step'] = window.steps
    req['type'] = 'fc'

    cached_file = f"wind_det_{levelist}_{window.name}.grb"
    print(req)
    common.fdb_read_to_file(cfg.fdb, req, cached_file)

    return cached_file


def fdb2cache_request_ens(cfg, levelist, window):
    """
    Retrieve vorticity and divergence from FDB and dump it to disk
    Ensemble forecast (control + perturbed)
    """

    req = cfg.request.copy()
    req.pop('stream_det')
    req['stream'] = req.pop('stream_ens')
    req["date"] = cfg.date.strftime("%Y%m%d")
    req["time"] = cfg.date.strftime("%H")+'00'
    req['levelist'] = levelist
    req['step'] = window.steps

    req_cf = req.copy()
    req_cf['type'] = 'cf'

    cached_file = f"wind_ens_{levelist}_{window.name}.grb"
    print(req)
    common.fdb_read_to_file(cfg.fdb, req_cf, cached_file)

    req_pf = req.copy()
    req_pf['type'] = 'pf'
    req_pf['number'] = range(1, cfg.members+1)
    common.fdb_read_to_file(cfg.fdb, req_pf, cached_file, mode='ab')

    return cached_file


def mir_wind(cfg, cached_file):
    """
    Compute wind components from cached grib file
    The grib file contains the vorticity and the divergence
    returns a list of messages containing the two components of velocity
    """

    interp_keys = cfg.interpolation_keys

    out = BytesIO()
    inp = mir.MultiDimensionalGribFileInput(cached_file, 2)

    job = mir.Job(vod2uv="1", **interp_keys)
    job.execute(inp, out)

    out.seek(0)
    reader = eccodes.StreamReader(out)
    messages = list(reader)

    return messages


def wind_norm_det(cfg, levelist, window):
    """
    Calculate deterministic (type=fc) wind speed
    """
    cached_file = fdb2cache_request_det(cfg, levelist, window)

    messages = mir_wind(cfg, cached_file)

    template = messages[0]

    steps = list(set([m["step"] for m in messages]))
    wind_paramids = list(set([m["paramId"] for m in messages]))
    assert len(wind_paramids) == 2
    u = np.asarray([m.get_array("values") for m in messages if m["paramId"] == wind_paramids[0]])
    v = np.asarray([m.get_array("values") for m in messages if m["paramId"] == wind_paramids[1]])

    wind_speed = np.sqrt(u * u + v * v)
    wind_speed = dict(zip(steps, wind_speed))

    return wind_speed, template


def wind_mean_std_eps(cfg, levelist, window):
    """
    Calculate ensemble (type=cf/pf) mean and standard deviation of wind speed
    """
    cached_file = fdb2cache_request_ens(cfg, levelist, window)

    messages = mir_wind(cfg, cached_file)

    template = messages[0]

    steps = list(set([m["step"] for m in messages]))
    wind_paramids = list(set([m["paramId"] for m in messages]))
    assert len(wind_paramids) == 2

    u = {step:[] for step in steps}
    v = {step:[] for step in steps}
    for m in messages:
        step = m['step']
        param = m["paramId"]
        if param == wind_paramids[0]:
            u[step].append(m.get_array("values"))
        elif param == wind_paramids[1]:
            v[step].append(m.get_array("values"))
        else:
            raise ValueError(f"Wrong paramId in message: {param}")

    u = np.asarray(list(u.values()))
    v = np.asarray(list(v.values()))

    ws = np.sqrt(u * u + v * v)
    mean = np.mean(ws, axis=0)
    mean = dict(zip(steps, mean))
    stddev = np.std(ws, axis=0)
    stddev = dict(zip(steps, stddev))

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

        self.request = self.options['request']
        self.windows = self.options['windows']
        self.levelist = self.options['levelist']
        self.interpolation_keys = self.options['interpolation_keys']


def main(args=None):

    parser = common.default_parser('Calculate wind speed mean/standard deviation')
    args = parser.parse_args(args)
    cfg = ConfigExtreme(args)

    for levelist in cfg.levelist:
        for window_options in cfg.windows:

            window = common.Window(window_options, include_init=True)

            # calculate wind speed for type=fc (deterministic)
            det, template_det = wind_norm_det(cfg, levelist, window)
            for step in window.steps:
                det_file = os.path.join(cfg.out_dir, f'det_{levelist}_{window.name}_{step}.grib')
                target_det = common.target_factory(cfg.target, out_file=det_file, fdb=cfg.fdb)
                template_det = template_det
                common.write_grib(target_det, template_det, det[step])

            # calculate mean/stddev of wind speed for type=pf/cf (eps)
            mean, std, template_ens = wind_mean_std_eps(cfg, levelist, window)
            for step in window.steps:
                mean_file = os.path.join(cfg.out_dir, f'mean_{levelist}_{window.name}_{step}.grib')
                target_mean = common.target_factory(cfg.target, out_file=mean_file, fdb=cfg.fdb)
                template_mean = template_ensemble(cfg, template_ens, 'em')
                common.write_grib(target_mean, template_mean, mean[step])
                std_file = os.path.join(cfg.out_dir, f'std_{levelist}_{window.name}_{step}.grib')
                target_std = common.target_factory(cfg.target, out_file=std_file, fdb=cfg.fdb)
                template_std = template_ensemble(cfg, template_ens, 'es')
                common.write_grib(target_std, template_std, std[step])


if __name__ == "__main__":
    import sys

    main(sys.argv)
