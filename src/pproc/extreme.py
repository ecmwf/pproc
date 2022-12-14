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
import yaml

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


def fdb_request_forecast(fc_keys, paramid, date, steps, member):

    req = fc_keys.copy()
    req["date"] = date.strftime("%Y%m%d")
    req["time"] = date.strftime("%H")+'00'
    req["step"] = steps
    req["param"] = paramid
    if member == 0:
        req['type'] = 'cf'
    else:
        req['type'] = 'pf'
        req['number'] = member

    return req


@dataclass
class Parameter(): # change id to paramid
    def __init__(self, paramid, options, ymdh):

        self.paramid = paramid
        self.name = options['name']
        self.clim_keys = options['clim_keys']
        self.fc_keys = options['fc_keys']
        self.out_keys = options['out_keys']
        params = self.fc_keys.pop('param')
        self.req_paramids = set(params) if isinstance(params, list) else {params}
        self.eps = float(options['eps'])
        self.sot = options['sot']
        self.type = options['preprocessing']
        window = options['windows']
        self.steps = list(range(window['start_step']+window['step_by'], window['end_step']+window['step_by'], window['step_by']))
        window_size = window['end_step']-window['start_step']
        self.suffix = f"{self.name}_{window_size:0>3}_{window['start_step']:0>3}h_{window['end_step']:0>3}h"
        self.suffix_ref = f"{self.name}{window_size:0>3}_{ymdh}_{window['start_step']:0>3}h_{window['end_step']:0>3}h"
        self.window = f"{window['start_step']}-{window['end_step']}"

    def accumulation(self, fields):
        if isinstance(fields, dict):
            vals = fields[self.paramid].values
        else:
            vals = fields
        if self.type == 'mean':
            nsteps = vals.shape[0]
            accum = np.sum(vals, axis=0) / nsteps
        elif self.type == 'min':
            accum = np.min(vals, axis=0)
        elif self.type == 'max':
            accum = np.max(vals, axis=0)
        else:
            raise Exception(f'Accumulation {self.type} not supported! Accepted values: (mean, min, max)')
        return accum

    def retrieve_fields(self, cfg, member):
        fields = {}
        for paramid in self.req_paramids:
            req = fdb_request_forecast(self.fc_keys, paramid, cfg.fc_date, self.steps, member)
            print(req)
            fields[paramid] = common.fdb_read(cfg.fdb, req)
        return fields
    
    def template(self, fields):
        return fields[self.paramid].attrs['grib_template']

    def preprocessing(self, cfg, member):
        fields = self.retrieve_fields(cfg, member)
        acc = self.accumulation(fields)
        return acc, self.template(fields)


class ParameterNorm(Parameter):
    def __init__(self, paramid, options, ymdh):
        super().__init__(paramid, options, ymdh)
        self.type = options['accumulation']
        assert len(self.req_paramids) == 2

    def template(self, fields):
        return list(fields.values())[0].attrs['grib_template']

    def accumulation(self, fields):
        u = list(fields.values())[0]
        v = list(fields.values())[1]
        norm = np.sqrt(u*u+v*v)
        return super().accumulation(norm.values)


class ParameterAccumulated(Parameter):
    def __init__(self, paramid, options, ymdh):
        super().__init__(paramid, options, ymdh)
        window = options['windows']
        self.steps = [window['start_step'], window['end_step']]
    
    def accumulation(self, fields):
        vals = fields[self.paramid].values
        return vals[-1]-vals[0]


class ParameterTreshold(Parameter):

    def __init__(self, paramid, options, ymdh):
        super().__init__(paramid, options, ymdh)
        self.type = options['accumulation']
        self.paramid_filter = options['fc_keys'].pop('paramid_filter', self.paramid)
        self.threshold = options['threshold']
        self.req_paramids.add(self.paramid_filter)

    def accumulation(self, fields):
        vals = fields[self.paramid].values
        vals_filt = fields[self.paramid_filter].values
        vals[vals_filt<=self.threshold] = 0
        return super().accumulation(vals)


def parameter_factory(parameters_options, ymdh):

    parameters = []
    for paramid, options in parameters_options.items():
        if options['preprocessing'] in ['min', 'max', 'mean']:
            param = Parameter(paramid, options, ymdh)
        elif options['preprocessing'] in ['accumulated']:
            param = ParameterAccumulated(paramid, options, ymdh)
        elif options['preprocessing'] in ['norm']:
            param = ParameterNorm(paramid, options, ymdh)
        elif options['preprocessing'] in ['threshold']:
            param = ParameterTreshold(paramid, options, ymdh)
        else:
            raise ValueError(f"Parameter preprocessing {options['preprocessing']} not supported")
        parameters.append(param)
    return parameters


def compute_forecast_operation(cfg, param):

    avg = []
    for member in range(cfg.members):
        acc, grib_template = param.preprocessing(cfg, member)
        avg.append(acc)
    avg = np.asarray(avg)
    print(f'Array computed from FDB: {avg.shape}')

    return avg, grib_template


def read_clim(cfg, param, n_clim=101):

    req = param.clim_keys.copy()
    req["date"] = cfg.clim_date.strftime("%Y%m%d")
    req["time"] = '0000'
    req["quantile"] = ['{}:100'.format(i) for i in range(n_clim)]
    req["step"] = f'{param.window}'

    da_clim = common.fdb_read(cfg.fdb, req)
    print(da_clim)

    return np.asarray(da_clim.values), da_clim.attrs['grib_template']


# def check_results(cfg, param):

#     check = True

#     print('Checking sot results')
#     for perc in param.sot:
#         test_sot = read_grib_file(os.path.join(cfg.out_dir, f'sot{perc}_{param.suffix}.grib'))
#         ref_sot = read_grib_file(os.path.join(cfg.ref_dir, f'sot{perc}_{param.suffix_ref}.grib'))
#         check = check and compare_arrs(test_sot, ref_sot)
#     print(check)
#     print('Checking efi results')
#     test_efi = read_grib_file(os.path.join(cfg.out_dir, f'efi_{param.suffix}.grib'))
#     ref_efi = read_grib_file(os.path.join(cfg.ref_dir, f'efi_{param.suffix_ref}.grib'))
#     print(test_efi.shape, ref_efi.shape)
#     check = check and compare_arrs(test_efi, ref_efi[0])
#     print(check)

#     return check


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        print(yaml.dump(self.options))
        self.fc_date = datetime.strptime(str(self.options['fc_date']), "%Y%m%d%H")
        ymdh = self.fc_date.strftime("%Y%m%d%H")

        self.parameters = parameter_factory(self.options['parameters'], ymdh)

        self.members = self.options['members']
        self.fdb = pyfdb.FDB()
        
        self.root_dir = self.options['root_dir']
        self.ref_dir = os.path.join(self.root_dir, 'efi', self.fc_date.strftime("%Y%m%d%H"))
        self.out_dir = os.path.join(self.root_dir, 'efi_test', self.fc_date.strftime("%Y%m%d%H"))

        self.clim_date = self.options.get('clim_date', climatology_date(self.fc_date))
        self.clim_dir = os.path.join(self.root_dir, 'clim', self.clim_date.strftime("%Y%m%d"))

        self.target = self.options['target']

        print(f'Forecast date is {self.fc_date}')
        print(f'Climatology date is {self.clim_date}')
        print(f'Parameters are {self.parameters}')
        print(f'Root directory is {self.root_dir}')


def extreme_template(cfg, param, template, template_clim):

    template_ext = template.copy()

    for key, value in param.out_keys.items():
        template_ext[key] = value
    
    # EFI specific stuff
    template_ext['stepRange'] = param.window
    if int(template_ext['timeRangeIndicator']) == 3:
        template_ext['numberIncludedInAverage'] = len(param.steps)
        template_ext['numberMissingFromAveragesOrAccumulations'] = 0        

    # set clim keys
    clim_keys = [
        'powerOfTenUsedToScaleClimateWeight',
        'weightAppliedToClimateMonth1',
        'firstMonthUsedToBuildClimateMonth1',
        'lastMonthUsedToBuildClimateMonth1',
        'firstMonthUsedToBuildClimateMonth2',
        'lastMonthUsedToBuildClimateMonth2',
        'numberOfBitsContainingEachPackedValue'
    ]
    for key in clim_keys:
        template_ext[key] = template_clim[key]

    # set fc keys
    fc_keys = [
        'date',
        'subCentre',
        'totalNumber',
    ]
    for key in fc_keys:
        template_ext[key] = template[key]
    
    return template_ext


def efi_template(template):
    template_efi = template.copy()
    template_efi['marsType'] = 27
    template_efi['efiOrder'] = 0
    template_efi['number'] = 0
    return template_efi

def efi_template_control(template):
    template_efi = template.copy()
    template_efi['marsType'] = 28
    template_efi['efiOrder'] = 0
    template_efi['totalNumber'] = 1
    template_efi['number'] = 0
    return template_efi

def sot_template(template, sot):
    template_sot = template.copy()
    template_sot['marsType'] = 38
    template_sot['number'] = sot
    if sot == 90:
        template_sot['efiOrder'] = 99
    elif sot == 10:
        template_sot['efiOrder'] = 1
    else:
        raise Exception("SOT value '{sot}' not supported in template! Only accepting 10 and 90")
    return template_sot


def main(args=None):

    parser = common.default_parser('Compute EFI and SOT from forecast and climatology for one parameter')
    args = parser.parse_args(args)
    cfg = ConfigExtreme(args)

    for param in cfg.parameters:

        fc_avg, template = compute_forecast_operation(cfg, param)
        print(f'Resulting averaged array: {fc_avg.shape}')

        clim, template_clim = read_clim(cfg, param)
        print(f'Climatology array: {clim.shape}')

        template_extreme = extreme_template(cfg, param, template, template_clim)

        print('Computing efi for the control member')
        efi_control = extreme.efi(clim, fc_avg, param.eps)
        out_file = os.path.join(cfg.out_dir, f'efi_control_{param.suffix}.grib')
        target = common.target_factory(cfg.target, out_file=out_file, fdb=cfg.fdb)
        template_efi = efi_template_control(template_extreme)
        common.write_grib(target, template_efi, efi_control)

        print('Computing efi')
        efi = extreme.efi(clim, fc_avg, param.eps)
        out_file = os.path.join(cfg.out_dir, f'efi_{param.suffix}.grib')
        target = common.target_factory(cfg.target, out_file=out_file, fdb=cfg.fdb)
        template_efi = efi_template(template_extreme)
        common.write_grib(target, template_efi, efi)

        sot = {}
        for perc in param.sot:
            print(f'Computing sot {perc}')
            sot[perc] = extreme.sot(clim, fc_avg, perc, param.eps)
            out_file = os.path.join(cfg.out_dir, f'sot{perc}_{param.suffix}.grib')
            target = common.target_factory(cfg.target, out_file=out_file, fdb=cfg.fdb)
            template_sot = sot_template(template_extreme, perc)
            common.write_grib(target, template_sot, sot[perc])

        cfg.fdb.flush()

        # if check_results(cfg, param):
        #     return 0
        # else:
        #     return 1


if __name__ == "__main__":
    main()
