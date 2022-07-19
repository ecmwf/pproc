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
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List

import eccodes
import pyfdb
from meteokit import extreme
from pproc import common


def climatology_date(fc_date):

    weekday = fc_date.weekday()

    # friday to monday -> take previous monday clim, else previous thursday clim
    if weekday == 0 or weekday > 3:
        clim_date = fc_date - timedelta(days=(weekday+4)%7)
    else:
        clim_date = fc_date - timedelta(days=weekday)

    return clim_date


def compare_arrs(dev, ref):
    dev = dev.flatten()
    ref = ref.flatten()
    dev = dev[np.isfinite(dev)]
    ref = ref[np.isfinite(ref)]
    if np.allclose(dev, ref, rtol=1e-4):
        print("OK")
        return True
    else:
        mask = np.logical_not(np.isclose(dev, ref, rtol=1e-4))
        print(mask.shape)
        print(dev[mask], ref[mask])
        max_diff = (np.abs(dev - ref)/max(ref)).max()
        print("{}/{} values differ, max rel diff {}".format(
            np.sum(mask), dev.size, max_diff))
        if max_diff > 1e-3:
            return False
        else:
            return True


def read_grib_file(in_file):
    reader = eccodes.FileReader(in_file)

    data = []
    for message in reader:
        data_array = message.get_array("values")

        # handle missing values and replace by nan
        if message.get('bitmapPresent'):
            missing = message.get('missingValue')
            data_array[data_array == missing] = np.nan

        data.append(data_array)

    return np.asarray(data)


def write_grib(template, data, out_dir, out_name):

    message = template.copy()

    # replace missing values if any
    missing = -9999
    is_missing = np.isnan(data).any()
    if is_missing:
        data[np.isnan(data)] = missing
        message.set('missingValue', missing)
        message.set('bitmapPresent', 1)
        
    message.set_array('values', data)

    if is_missing:
        n_missing1 = len(data[data==missing])
        n_missing2 = message.get('numberOfMissing')
        if n_missing1 != n_missing2:
            raise Exception(f'Number of missing values in the message not consistent, is {n_missing1} and should be {n_missing2}')

    out_file = os.path.join(out_dir, out_name)
    with open(out_file,"ab") as outfile:
    	message.write_to(outfile)


def fdb_request_forecast(stream, paramid, date, steps, member):

    req = {
            "class": "od",
            "expver": "0001",
            "stream": stream,
            "date": date.strftime("%Y%m%d"),
            "time": date.strftime("%H")+'00',
            "domain": "g",
            "type": "cf",
            "levtype": "sfc",
            "step": steps,
            "param": paramid
    }

    if member == 0:
        req['type'] = 'cf'
    else:
        req['type'] = 'pf'
        req['number'] = member

    return req


@dataclass
class Parameter(): # change id to paramid
    name: str
    id: int
    clim_id: int = id
    efi_id: int = id
    eps: float = -1e-4
    sot: List = field(default_factory=lambda: [90])
    stream: str = 'enfo'
    stream_clim: str = 'efhs'
    accumulation: str = 'mean'

    def compute_accumulation(self, fields):
        if self.accumulation == 'mean':
            nsteps = fields.shape[0]
            accum = np.sum(fields, axis=0) / nsteps
        elif self.accumulation == 'min':
            accum = np.min(fields, axis=0)
        elif self.accumulation == 'max':
            accum = np.max(fields, axis=0)
        else:
            raise Exception(f'Accumulation {self.accumulation} not supported! Accepted values: (mean, min, max)')
        return accum

    def preprocessing(self, cfg, member, steps):
        fc_date = cfg.fc_date
        req = fdb_request_forecast(self.stream, self.id, fc_date, steps[1:], member)
        da = common.fdb_read(cfg.fdb, req)

        acc = self.compute_accumulation(da.values)
        return acc, da.attrs['grib_template']


class ParameterVector(Parameter):
    def __init__(self, *args, **kwargs):
        id1 = kwargs.pop('id1')
        id2 = kwargs.pop('id2')
        super().__init__(*args, **kwargs)
        self.id1 = id1
        self.id2 = id2
    
    def preprocessing(self, cfg, member, steps):
        fc_date = cfg.fc_date

        req_1 = fdb_request_forecast(self.stream, self.id1, fc_date, steps[1:], member)
        da_1 = common.fdb_read(cfg.fdb, req_1)
        req_2 = fdb_request_forecast(self.stream, self.id2, fc_date, steps[1:], member)
        da_2 = common.fdb_read(cfg.fdb, req_2)

        norm = np.sqrt(da_1*da_1+da_2*da_2)

        acc = self.compute_accumulation(norm.values)
        return acc, da_1.attrs['grib_template']


class ParameterDifference(Parameter):
    
    def preprocessing(self, cfg, member, steps):
        fc_date = cfg.fc_date

        req = fdb_request_forecast(self.stream, self.id, fc_date, [steps[0], steps[-1]], member)
        da = common.fdb_read(cfg.fdb, req)
        vals = da.values

        acc = vals[1]-vals[0]
        return acc, da.attrs['grib_template']

class ParameterCape(Parameter):

    def __init__(self, *args, **kwargs):
        threshold = kwargs.pop('threshold')
        super().__init__(*args, **kwargs)
        self.threshold = threshold
    
    def preprocessing(self, cfg, member, steps):
        fc_date = cfg.fc_date

        req = fdb_request_forecast(self.stream, self.id, fc_date, steps[1:], member)
        da = common.fdb_read(cfg.fdb, req)
        vals = da.values

        vals[vals<=self.threshold] = 0

        acc = self.compute_accumulation(vals)
        return acc, da.attrs['grib_template']


class ParameterCapeShear(Parameter):

    def __init__(self, *args, **kwargs):
        threshold = kwargs.pop('threshold')
        id_filt = kwargs.pop('id_filt')
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.id_filt = id_filt
    
    def preprocessing(self, cfg, member, steps):
        fc_date = cfg.fc_date

        req = fdb_request_forecast(self.stream, self.id, fc_date, steps, member)
        da = common.fdb_read(cfg.fdb, req)
        vals = da.values

        req_filt = fdb_request_forecast(self.stream, self.id_filt, fc_date, steps[1:], member)
        da_filt = common.fdb_read(cfg.fdb, req_filt)
        vals_filt = da_filt.values

        vals[vals_filt<=self.threshold] = 0

        acc = self.compute_accumulation(vals)
        return acc, da.attrs['grib_template']


def parameter_factory(paramid):

    if paramid == 167:
        parameter = Parameter('2t', 167, clim_id=228004, sot=[10, 90])
    elif paramid == 122:
        parameter = Parameter('2tmin', 122, clim_id=202, efi_id=202, sot=[10, 90], accumulation='min')
    elif paramid == 121:
        parameter = Parameter('2tmax', 121, clim_id=201, efi_id=201, sot=[10, 90], accumulation='max')
    elif paramid == 207:
        parameter = Parameter('10ff', 207, clim_id=228005)
    elif paramid == 123:
        parameter = Parameter('10fg', 123, clim_id=49, efi_id=49, accumulation='max')
    elif paramid == 229:
        parameter = Parameter('hsttmax', 229, clim_id=140200, efi_id=216, stream='waef', stream_clim='wehs', accumulation='max')
    elif paramid == 228:
        parameter = ParameterDifference('tp', 228, clim_id=228, efi_id=228, eps=1e-4)
    elif paramid == 144:
        parameter = ParameterDifference('sf', 144, clim_id=144, efi_id=144, eps=1e-4)
    elif paramid == 162071:
        parameter = ParameterVector('wvf', 162045, id1=162071, id2=162072, clim_id=162045, efi_id=45)
    elif paramid == 228035:
        parameter = ParameterCape('cape', 228035, clim_id=59, efi_id=59, eps=1e-4, accumulation='max', threshold=10)
    elif paramid == 228036:
        parameter = ParameterCapeShear('capeshear', 228036, id_filt=228035, clim_id=228044, efi_id=44, eps=1e-4, accumulation='max', threshold=10)
    else:
        raise Exception(f'Param ID {paramid} not supported')

    return parameter


def compute_forecast_operation(cfg):

    fc_date = cfg.fc_date
    ymdh = fc_date.strftime("%Y%m%d%H")

    # read reference file
    ref_dir = os.path.join(cfg.root_dir, 'prepeps', ymdh)
    avg_ref = read_grib_file(os.path.join(ref_dir, f'eps_{cfg.suffix_ref}.grib'))
    print(f'Reference array: {avg_ref.shape}')

    avg = []
    for member in range(cfg.members):
        acc, grib_template = cfg.parameter.preprocessing(cfg, member, cfg.steps[0])
        avg.append(acc)

    avg = np.asarray(avg)
    print(f'Array computed from FDB: {avg.shape}')
    if not compare_arrs(avg, avg_ref):
        exit(1)

    return avg_ref, grib_template


def read_clim(cfg, n_clim=101):

    clim_ymd = cfg.clim_date.strftime("%Y%m%d")

    req = {
        'class': 'od',
        'expver': '0001',
        'stream': cfg.parameter.stream_clim,
        'date': clim_ymd,
        'time': '0000',
        'domain': 'g',
        'type': 'cd',
        'levtype': 'sfc',
        'quantile': ['{}:100'.format(i) for i in range(n_clim)],
        'step': f'{cfg.steps[0][0]}-{cfg.steps[0][-1]}',
        'param': cfg.parameter.clim_id
    }
    da_clim = common.fdb_read(cfg.fdb, req)
    print(da_clim)

    return np.asarray(da_clim.values)


def compute_efi(cfg, fc_avg, clim, template):

    efi = extreme.efi(clim, fc_avg, cfg.parameter.eps)

    write_grib(template, efi, cfg.out_dir, f'efi_{cfg.suffix}.grib')

    return efi


def compute_sot(cfg, fc_avg, clim, template):

    sot = {}
    for perc in cfg.parameter.sot:
        sot[perc] = extreme.sot(clim, fc_avg, perc, cfg.parameter.eps)
        write_grib(template, sot[perc], cfg.out_dir, f'sot{perc}_{cfg.suffix}.grib')

    return sot


def check_results(cfg):

    check = True

    print('Checking sot results')
    for perc in cfg.parameter.sot:
        test_sot = read_grib_file(os.path.join(cfg.out_dir, f'sot{perc}_{cfg.suffix}.grib'))
        ref_sot = read_grib_file(os.path.join(cfg.ref_dir, f'sot{perc}_{cfg.suffix_ref}.grib'))
        check = check and compare_arrs(test_sot, ref_sot)
    print(check)
    print('Checking efi results')
    test_efi = read_grib_file(os.path.join(cfg.out_dir, f'efi_{cfg.suffix}.grib'))
    ref_efi = read_grib_file(os.path.join(cfg.ref_dir, f'efi_{cfg.suffix_ref}.grib'))
    print(test_efi.shape, ref_efi.shape)
    check = check and compare_arrs(test_efi, ref_efi[0])
    print(check)

    return check


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.fc_date = datetime.strptime(str(self.options['fc_date']), "%Y%m%d%H")

        self.parameter = parameter_factory(self.options['paramid'])

        self.steps = []
        for steps in self.options['steps']:
            start_step = int(steps['start_step'])
            end_step = int(steps['end_step'])
            timestep = int(steps['timestep'])
            self.steps.append(list(range(start_step, end_step+timestep, timestep)))

        self.members = self.options['members']
        self.fdb = pyfdb.FDB()
        
        self.root_dir = self.options['root_dir']
        self.ref_dir = os.path.join(self.root_dir, 'efi', self.fc_date.strftime("%Y%m%d%H"))
        self.out_dir = os.path.join(self.root_dir, 'efi_test', self.fc_date.strftime("%Y%m%d%H"))

        self.clim_date = self.options.get('clim_date', climatology_date(self.fc_date))
        self.clim_dir = os.path.join(self.root_dir, 'clim', self.clim_date.strftime("%Y%m%d"))

        window_size = self.steps[0][-1]-self.steps[0][0]
        ymdh = self.fc_date.strftime("%Y%m%d%H")
        self.suffix = f"{self.parameter.name}_{window_size:0>3}_{self.steps[0][0]:0>3}h_{self.steps[0][-1]:0>3}h"
        self.suffix_ref = f"{self.parameter.name}{window_size:0>3}_{ymdh}_{self.steps[0][0]:0>3}h_{self.steps[0][-1]:0>3}h"

        print(f'Forecast date is {self.fc_date}')
        print(f'Climatology date is {self.clim_date}')
        print(f'ParamID is {self.parameter.id}')
        print(f'steps are {self.steps}')
        print(f'Root directory is {self.root_dir}')


def main(args=None):

    parser = common.default_parser('Compute EFI and SOT from forecast and climatology for one parameter')
    args = parser.parse_args(args)
    cfg = ConfigExtreme(args)

    fc_avg, template = compute_forecast_operation(cfg)
    print(f'Resulting averaged array: {fc_avg.shape}')

    clim = read_clim(cfg)
    print(f'Climatology array: {clim.shape}')

    efi = compute_efi(cfg, fc_avg, clim, template)

    sot = compute_sot(cfg, fc_avg, clim, template)

    if check_results(cfg):
        return 0
    else:
        return 1


if __name__ == "__main__":
    main()
