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


def fdb_request_det(cfg, levelist, window):
    """
    Retrieve vorticity and divergence from FDB and dump it to disk
    Deterministic forecast
    """

    req = cfg.request.copy()
    req.pop('stream_ens')
    req['stream'] = req.pop('stream_det')
    req["date"] = cfg.date.strftime("%Y%m%d")
    req["time"] = cfg.date.strftime("%H")+'00'
    if req['levtype'] != 'sfc':
        req['levelist'] = levelist
    req['step'] = window.steps
    req['type'] = 'fc'

    if cfg.vod2uv:
        cached_file = f"wind_det_{levelist}_{window.name}.grb"
        print(req)
        common.fdb_read_to_file(cfg.fdb, req, cached_file)
        messages = mir_wind(cfg, cached_file)
    else:
        out =  common.fdb_retrieve(cfg.fdb, req, cfg.interpolation_keys)
        reader = eccodes.StreamReader(out)
        messages = list(reader)

    return messages


def fdb_request_ens(cfg, levelist, window):
    """
    Retrieve vorticity and divergence from FDB and dump it to disk
    Ensemble forecast (control + perturbed)
    """

    req = cfg.request.copy()
    req.pop('stream_det')
    req['stream'] = req.pop('stream_ens')
    req["date"] = cfg.date.strftime("%Y%m%d")
    req["time"] = cfg.date.strftime("%H")+'00'
    if req['levtype'] != 'sfc':
        req['levelist'] = levelist
    req['step'] = window.steps

    req_cf = req.copy()
    req_cf['type'] = 'cf'

    if cfg.vod2uv:
        cached_file = f"wind_ens_{levelist}_{window.name}.grb"
        print(req)
        common.fdb_read_to_file(cfg.fdb, req_cf, cached_file)

        req_pf = req.copy()
        req_pf['type'] = 'pf'
        req_pf['number'] = range(1, cfg.members+1)
        common.fdb_read_to_file(cfg.fdb, req_pf, cached_file, mode='ab')
        messages = mir_wind(cfg, cached_file)
    else:
        out =  common.fdb_retrieve(cfg.fdb, req_cf, cfg.interpolation_keys)
        reader = eccodes.StreamReader(out)
        messages = list(reader)

        req_pf = req.copy()
        req_pf['type'] = 'pf'
        req_pf['number'] = range(1, cfg.members+1)
        out =  common.fdb_retrieve(cfg.fdb, req_pf, cfg.interpolation_keys)
        reader = eccodes.StreamReader(out)
        messages += list(reader)

    return messages


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


def wind_speed_det(cfg, levelist, window):
    """
    Calculate deterministic (type=fc) wind speed
    """
    messages = fdb_request_det(cfg, levelist, window)

    template = messages[0]

    steps = list(set([m["step"] for m in messages]))
    wind_paramids = list(set([m["paramId"] for m in messages]))
    assert len(wind_paramids) == 2
    u = np.asarray([m.get_array("values") for m in messages if m["paramId"] == wind_paramids[0]])
    v = np.asarray([m.get_array("values") for m in messages if m["paramId"] == wind_paramids[1]])

    wind_speed = np.sqrt(u * u + v * v)
    wind_speed = dict(zip(steps, wind_speed))

    return wind_speed, template


def wind_speed_eps(cfg, levelist, window):
    """
    Calculate ensemble (type=cf/pf) mean and standard deviation of wind speed
    """
    messages = fdb_request_ens(cfg, levelist, window)

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
    ws = dict(zip(steps, ws))

    return ws, template

def mean_and_std(ws):
    mean = {}
    stddev = {}
    for step, values in ws.items():
        mean[step] = np.mean(values, axis=1)
        stddev[step] = np.std(values, axis=1)
    return mean, stddev

def template_eps_speed(cfg, template, step, number):
    template_eps = template.copy()
    template_eps.set('bitsPerValue', 24)
    template_eps.set("step", step)
    if number == 0:
        template_eps.set('type', 'cf')
    else:
        template_eps.set({
            'type': 'pf',
            'number': number
        })
    if step == 0:
        template_eps.set('timeRangeIndicator', 1)
    else:
        template_eps.set('timeRangeIndicator', 0)
    for key, value in cfg.options['grib_set'].items():
        template_eps.set(key, value)
    return template_eps

def template_ensemble(cfg, template, step, marstype):
    template_ens = template.copy()
    template_ens.set('bitsPerValue', 24)
    template_ens.set("marsType", marstype)
    template_ens.set("step", step)
    if step == 0:
        template_ens.set('timeRangeIndicator', 1)
    else:
        template_ens.set('timeRangeIndicator', 0)
    for key, value in cfg.options['grib_set'].items():
        template_ens.set(key, value)
    return template_ens


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.date = datetime.strptime(str(self.options['fc_date']), "%Y%m%d%H")
        self.root_dir = self.options['root_dir']
        self.target = self.options['target']
        self.out_dir = os.path.join(self.root_dir, self.date.strftime("%Y%m%d%H"))

        self.fdb = pyfdb.FDB()

        self.request = self.options['request']
        self.windows = self.options['windows']
        self.levelist = self.options.get('levelist', [0])
        self.interpolation_keys = self.options.get('interpolation_keys', None)
        self.vod2uv = self.options.get('vod2uv', False)

        if args.eps_ws or args.eps_mean_std:
            self.members = self.options['members']


def main(args=None):

    parser = common.default_parser('Calculate wind speed')
    parser.add_argument('--det_ws', action='store_true', default=False, help='Wind speed for type=fc')
    parser.add_argument('--eps_ws', action='store_true', default=False, help='Wind speed for type=pf/cf')
    parser.add_argument('--eps_mean_std', action='store_true', default=False, help='Wind speed mean/std for type=pf/cf')
    args = parser.parse_args()
    cfg = ConfigExtreme(args)

    for levelist in cfg.levelist:
        for window_options in cfg.windows:

            window = common.Window(window_options, include_init=True)

            # calculate wind speed for type=fc (deterministic)
            if args.det_ws:
                det, template_det = wind_speed_det(cfg, levelist, window)
                for step in window.steps:
                    det_file = os.path.join(cfg.out_dir, f'det_{levelist}_{window.name}_{step}.grib')
                    target_det = common.target_factory(cfg.target, out_file=det_file, fdb=cfg.fdb)
                    template_det = template_ensemble(cfg, template_det, step, 'fc')
                    common.write_grib(target_det, template_det, det[step])

            # calculate mean/stddev of wind speed for type=pf/cf (eps)
            if args.eps_ws or args.eps_mean_std:
                wind_speed, template_ens = wind_speed_eps(cfg, levelist, window)
                for step in window.steps: 
                    print(step)
                    if args.eps_ws:
                        for number in range(cfg.members + 1):
                            eps_file = os.path.join(cfg.out_dir, f'eps_{levelist}_{window.name}_{step}_{number}.grib')
                            target_eps = common.target_factory(cfg.target, out_file=eps_file, fdb=cfg.fdb)
                            template_eps = template_eps_speed(cfg, template_ens, step, number)
                            common.write_grib(target_eps, template_eps, wind_speed[step][number])
                    if args.eps_mean_std:
                        mean, std = mean_and_std(wind_speed)
                        mean_file = os.path.join(cfg.out_dir, f'mean_{levelist}_{window.name}_{step}.grib')
                        target_mean = common.target_factory(cfg.target, out_file=mean_file, fdb=cfg.fdb)
                        template_mean = template_ensemble(cfg, template_ens, step, 'em')
                        common.write_grib(target_mean, template_mean, mean[step])
                        std_file = os.path.join(cfg.out_dir, f'std_{levelist}_{window.name}_{step}.grib')
                        target_std = common.target_factory(cfg.target, out_file=std_file, fdb=cfg.fdb)
                        template_std = template_ensemble(cfg, template_ens, step, 'es')
                        common.write_grib(target_std, template_std, std[step])


if __name__ == "__main__":
    import sys

    main(sys.argv)
