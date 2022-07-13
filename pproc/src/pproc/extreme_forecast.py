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
import argparse
import numpy as np
from datetime import datetime, timedelta
import yaml

import eccodes
import pyfdb
from meteokit import extreme


def climatology_date(cfg):

    fc_date = cfg.fc_date
    weekday = fc_date.weekday()

    # friday to monday -> take previous monday clim, else previous thursday clim
    if weekday == 0 or weekday > 3:
        clim_date = fc_date - timedelta(days=(weekday+4)%7)
    else:
        clim_date = fc_date - timedelta(days=weekday)

    return clim_date


def compute_avg(fields):
    nsteps = fields.shape[0]
    return np.sum(fields, axis=0) / nsteps


def compare_arrs(dev, ref):
    dev = dev.flatten()
    ref = ref.flatten()
    dev = dev[np.isfinite(dev)]
    ref = ref[np.isfinite(ref)]
    if np.allclose(dev, ref, rtol=1e-4):
        print("OK")
        return 0
    else:
        mask = np.logical_not(np.isclose(dev, ref, rtol=1e-4))
        print(mask.shape)
        print(dev[mask], ref[mask])
        print("{}/{} values differ, max rel diff {}".format(
            np.sum(mask), dev.size, (np.abs(dev - ref) / ref).max()))
        return 1


def read_grib_fdb(fdb, req):
    fdb_reader = fdb.retrieve(req)
    eccodes_reader = eccodes.StreamReader(fdb_reader)

    data = []
    for message in eccodes_reader:
        data_array = message.get_array("values")

        # handle missing values and replace by nan
        if message.get('bitmapPresent'):
            missing = message.get('missingValue')
            data_array[data_array == missing] = np.nan

        data.append(data_array)

    return np.asarray(data)


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
    reader = eccodes.FileReader(template)
    messages = list(reader)
    message = messages[0]

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


def compute_forecast_average(cfg):

    fc_date = cfg.fc_date
    ymdh = fc_date.strftime("%Y%m%d%H")

    # read reference file
    ref_dir = os.path.join(cfg.root_dir, 'prepeps', ymdh)
    avg_ref = read_grib_file(os.path.join(ref_dir, f'eps_{cfg.param}{cfg.window_size:0>3}_{ymdh}_{cfg.window[0]:0>3}h_{cfg.window[1]:0>3}h.grib'))
    print(f'Reference array: {avg_ref.shape}')

    req = {
        "class": "od",
        "expver": "0001",
        "stream": "enfo",
        "date": fc_date.strftime("%Y%m%d"),
        "time": fc_date.strftime("%H")+'00',
        "domain": "g",
        "type": "cf",
        "levtype": "sfc",
        "step": [6, 12, 18, 24],
        "param": cfg.paramid,
    }

    avg = []
    for m in range(cfg.members):
        if m == 0:
            req['type'] = 'cf'
        else:
            req['type'] = 'pf'
            req['number'] = m

        vals = read_grib_fdb(cfg.fdb, req)
        avg_member = compute_avg(vals)
        avg.append(avg_member)

    avg = np.asarray(avg)
    print(f'Array computed from FDB: {avg.shape}')
    compare_arrs(avg, avg_ref)

    return avg_ref


def read_clim(cfg, n_clim=101):

    clim_ymd = cfg.clim_date.strftime("%Y%m%d")

    req = {
        'class': 'od',
        'expver': '0001',
        'stream': 'efhs',
        'date': clim_ymd,
        'time': '0000',
        'domain': 'g',
        'type': 'cd',
        'levtype': 'sfc',
        'step': f'{cfg.window[0]}-{cfg.window[1]}',
        'quantile': ['{}:100'.format(i) for i in range(n_clim)],
        'param': cfg.clim_id
    }
    vals_clim = read_grib_fdb(cfg.fdb, req)
    # vals_clim = read_grib_file(os.path.join(cfg.clim_dir, f'clim_{cfg.param}{cfg.window_size:0>3}_{clim_ymd}_{cfg.window[0]:0>3}h_{cfg.window[1]:0>3}h_perc.grib'))

    return np.asarray(vals_clim)


def compute_efi(cfg, fc_avg, clim):

    efi = extreme.efi(clim, fc_avg, cfg.eps)

    write_grib(cfg.template, efi, cfg.out_dir, f'efi_{cfg.param}_{cfg.window_size:0>3}_{cfg.window[0]:0>3}h_{cfg.window[1]:0>3}h.grib')

    return efi


def compute_sot(cfg, fc_avg, clim):

    sot = {}
    for perc in cfg.sot_values:
        sot[perc] = extreme.sot(clim, fc_avg, perc, cfg.eps)
        write_grib(cfg.template, sot[perc], cfg.out_dir, f'sot{perc}_{cfg.param}_{cfg.window_size:0>3}_{cfg.window[0]:0>3}h_{cfg.window[1]:0>3}h.grib')

    return sot


def check_results(cfg):

    check = 0

    fc_date = cfg.fc_date.strftime("%Y%m%d%H")

    print('Checking sot results')
    for perc in cfg.sot_values:
        test_sot = read_grib_file(os.path.join(cfg.out_dir, f'sot{perc}_{cfg.param}_{cfg.window_size:0>3}_{cfg.window[0]:0>3}h_{cfg.window[1]:0>3}h.grib'))
        ref_sot = read_grib_file(os.path.join(cfg.ref_dir, f'sot{perc}_{cfg.param}{cfg.window_size:0>3}_{fc_date}_{cfg.window[0]:0>3}h_{cfg.window[1]:0>3}h.grib'))
        check += compare_arrs(test_sot, ref_sot)

    print('Checking efi results')
    test_efi = read_grib_file(os.path.join(cfg.out_dir, f'efi_{cfg.param}_{cfg.window_size:0>3}_{cfg.window[0]:0>3}h_{cfg.window[1]:0>3}h.grib'))
    ref_efi = read_grib_file(os.path.join(cfg.ref_dir, f'efi_{cfg.param}{cfg.window_size:0>3}_{fc_date}_{cfg.window[0]:0>3}h_{cfg.window[1]:0>3}h.grib'))
    print(test_efi.shape, ref_efi.shape)
    check += compare_arrs(test_efi, ref_efi[0])

    return check


class Config():
    def __init__(self, args):

        self.fc_date = datetime.strptime(args.fc_date,"%Y%m%d%H")

        with open(args.parameter, 'r') as file:
            parameter = yaml.load(file, Loader=yaml.SafeLoader)
        
        self.param = parameter['name']
        self.paramid = parameter['param_id']
        self.clim_id = parameter['clim_id']
        self.efi_id = parameter['efi_id']
        self.eps = parameter['eps']
        self.sot_values = [int(i) for i in str(parameter['sot']).split(',')]

        self.window = [int(i) for i in args.window.split('-')]
        self.window_size = self.window[1]-self.window[0]
        self.window_step = 6

        self.template = args.template
        self.members = 51
        self.fdb = pyfdb.FDB()
        
        self.root_dir = args.root_dir
        self.ref_dir = os.path.join(self.root_dir, 'efi', self.fc_date.strftime("%Y%m%d%H"))
        self.out_dir = os.path.join(self.root_dir, 'efi_test', self.fc_date.strftime("%Y%m%d%H"))

        self.clim_date = climatology_date(self)
        self.clim_dir = os.path.join(self.root_dir, 'clim', self.clim_date.strftime("%Y%m%d"))

        print(f'Forecast date is {self.fc_date}')
        print(f'Climatology date is {self.clim_date}')
        print(f'Parameter is {self.param}')
        print(f'ParamID is {self.paramid}')
        print(f'window is {self.window}, size {self.window_size}')
        print(f'Root directory is {self.root_dir}')
        print(f'Grib template is {self.template}')
        print(f'eps is {self.eps}')


def main(args=None):

    parser = argparse.ArgumentParser(description='Small python EFI test')
    parser.add_argument('fc_date', help='Forecast date')
    parser.add_argument('parameter', help='Parameter file')
    parser.add_argument('window', help='Averaging window')
    parser.add_argument('root_dir', help='Root directory')
    parser.add_argument('template', help='GRIB template')

    args = parser.parse_args(args)
    cfg = Config(args)

    fc_avg = compute_forecast_average(cfg)
    print(f'Resulting averaged array: {fc_avg.shape}')

    clim = read_clim(cfg)
    print(f'Climatology array: {clim.shape}')

    efi = compute_efi(cfg, fc_avg, clim)

    sot = compute_sot(cfg, fc_avg, clim)

    return check_results(cfg)


if __name__ == "__main__":
    main()
