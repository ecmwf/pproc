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
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass

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


def read_grib_template(in_file):
    reader = eccodes.FileReader(in_file)
    return next(reader)


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
    def __init__(self, paramid, options, window):

        self.paramid = paramid
        self.window = window
        self.name = options['name']
        self.clim_keys = options['clim_keys']
        self.fc_keys = options['fc_keys']
        self.out_keys = options['out_keys']
        params = self.fc_keys.pop('param')
        self.req_paramids = set(params) if isinstance(params, list) else {params}
        self.eps = float(options['eps'])
        self.sot = [int(x) for x in options['sot']]
        self.type = options['preprocessing']
        self.suffix = f"{self.name}_{window.suffix}"
        self.steps = window.steps

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
            fields[paramid] = common.fdb_read(cfg.fdb, req)
        return fields
    
    def template(self, fields):
        return fields[self.paramid].attrs['grib_template']

    def preprocessing(self, cfg, member):
        fields = self.retrieve_fields(cfg, member)
        acc = self.accumulation(fields)
        return acc, self.template(fields)


class ParameterNorm(Parameter):
    def __init__(self, paramid, options, window):
        super().__init__(paramid, options, window)
        self.type = options['accumulation']
        assert len(self.req_paramids) == 2

    def template(self, fields):
        return list(fields.values())[0].attrs['grib_template']

    def accumulation(self, fields):
        u = list(fields.values())[0]
        v = list(fields.values())[1]
        norm = np.sqrt(u*u+v*v)
        return super().accumulation(norm.values)  # calling parent accumulation to compute mean, max, etc.


class ParameterAccumulated(Parameter):
    def __init__(self, paramid, options, window):
        super().__init__(paramid, options, window)
        self.steps = [window.start, window.end]
    
    def accumulation(self, fields):
        vals = fields[self.paramid].values
        return vals[-1]-vals[0]


class ParameterTreshold(Parameter):

    def __init__(self, paramid, options, window):
        super().__init__(paramid, options, window)
        self.type = options['accumulation']
        self.paramid_filter = options['fc_keys'].pop('paramid_filter', self.paramid)
        self.threshold = options['threshold']
        self.req_paramids.add(self.paramid_filter)

    def accumulation(self, fields):
        vals = fields[self.paramid].values
        vals_filt = fields[self.paramid_filter].values
        vals[vals_filt<=self.threshold] = 0
        return super().accumulation(vals)  # calling parent accumulation to compute mean, max, etc.


def parameter_factory(parameters_options):

    parameters = []
    for paramid, options in parameters_options.items():
        param_windows = []
        for window_options in options['windows']:
            window = common.Window(window_options, include_init=False)
            if options['preprocessing'] in ['min', 'max', 'mean']:
                param = Parameter(paramid, options, window)
            elif options['preprocessing'] in ['accumulated']:
                param = ParameterAccumulated(paramid, options, window)
            elif options['preprocessing'] in ['norm']:
                param = ParameterNorm(paramid, options, window)
            elif options['preprocessing'] in ['threshold']:
                param = ParameterTreshold(paramid, options, window)
            else:
                raise ValueError(f"Parameter preprocessing {options['preprocessing']} not supported")
            param_windows.append(param)
        parameters.append(param_windows)
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
    req["step"] = f'{param.window.name}'

    da_clim = common.fdb_read(cfg.fdb, req)
    da_clim_sorted = da_clim.reindex(quantile=[f'{x}:100' for x in range(n_clim)])
    print(da_clim_sorted)

    return np.asarray(da_clim_sorted.values), da_clim.attrs['grib_template']


def extreme_template(param, template_fc, template_clim):

    template_ext = template_fc.copy()

    for key, value in param.out_keys.items():
        template_ext[key] = value
    
    # EFI specific stuff
    template_ext['stepRange'] = param.window.name
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
        template_ext[key] = template_fc[key]
    
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


class ConfigExtreme(common.Config):
    def __init__(self, args):
        super().__init__(args)

        self.fc_date = datetime.strptime(str(self.options['fc_date']), "%Y%m%d%H")

        self.parameters = parameter_factory(self.options['parameters'])

        self.members = int(self.options['members'])
        self.fdb = pyfdb.FDB()
        
        self.root_dir = self.options['root_dir']
        self.out_dir = os.path.join(self.root_dir, 'efi_test', self.fc_date.strftime("%Y%m%d%H"))

        self.clim_date = self.options.get('clim_date', climatology_date(self.fc_date))

        self.target = self.options['target']

        print(f'Forecast date is {self.fc_date}')
        print(f'Climatology date is {self.clim_date}')
        print(f'Parameters are {self.parameters}')
        print(f'Root directory is {self.root_dir}')


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)

    parser = common.default_parser('Compute EFI and SOT from forecast and climatology for one parameter')
    args = parser.parse_args(args)
    cfg = ConfigExtreme(args)
    recovery = common.Recovery(cfg.root_dir, args.config, cfg.fc_date, args.recover)

    for param_windows in cfg.parameters:
        for param in param_windows:
            if recovery.existing_checkpoint(param.paramid, param.window.name):
                print(f'Recovery: skipping param {param.paramid} window {param.window.name}')
                continue

            fc_avg, template_fc = compute_forecast_operation(cfg, param)
            print(f'Resulting averaged array: {fc_avg.shape}')

            clim, template_clim = read_clim(cfg, param)
            print(f'Climatology array: {clim.shape}')

            template_extreme = extreme_template(param, template_fc, template_clim)

            print('Computing efi for the control member')
            efi_control = extreme.efi(clim, fc_avg[: 1], param.eps)
            template_efi = efi_template_control(template_extreme)
            
            out_file = os.path.join(cfg.out_dir, f'efi_control_{param.suffix}.grib')
            target = common.target_factory(cfg.target, out_file=out_file, fdb=cfg.fdb)
            common.write_grib(target, template_efi, efi_control)

            print('Computing efi')
            efi = extreme.efi(clim, fc_avg, param.eps)
            template_efi = efi_template(template_extreme)
            
            out_file = os.path.join(cfg.out_dir, f'efi_{param.suffix}.grib')
            target = common.target_factory(cfg.target, out_file=out_file, fdb=cfg.fdb)
            common.write_grib(target, template_efi, efi)

            sot = {}
            for perc in param.sot:
                print(f'Computing sot {perc}')
                
                sot[perc] = extreme.sot(clim, fc_avg, perc, param.eps)
                template_sot = sot_template(template_extreme, perc)
                
                out_file = os.path.join(cfg.out_dir, f'sot{perc}_{param.suffix}.grib')
                target = common.target_factory(cfg.target, out_file=out_file, fdb=cfg.fdb)
                common.write_grib(target, template_sot, sot[perc])

            recovery.add_checkpoint(param.paramid, param.window.name)

            cfg.fdb.flush()
    recovery.clean_file()

if __name__ == "__main__":
    main()
