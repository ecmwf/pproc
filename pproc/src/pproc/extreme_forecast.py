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
from dataclasses import dataclass

import eccodes
import pyfdb

from meteokit import extreme


@dataclass
class Parameter():
    name: str
    paramId: str
    eps: float = -1e-4


def build_parameter_database():

    params_db = dict(
        2t = Parameter('2t', '167.128'),
        2tmin = Parameter('2tmin', '122'),
        2tmax = Parameter('2tmax', '121'),
        10ff = Parameter('10ff', '165'),
        10fg = Parameter('10fg', '123'),
        10dd = Parameter('10dd', '165'),
        tcc = Parameter('tcc', '164'),
        500z = Parameter('500z', '129'),
        850t = Parameter('850t', '130'),
        hsttmax = Parameter('hsttmax', '229'),
        hsstmax = Parameter('hsstmax', '229'),
        wvf = Parameter('wvf', '162071'),
        tp = Parameter('tp', '228', 1e-4),
        ts = Parameter('ts', '144', 1e-4),
        sf = Parameter('sf', '144', 1e-4),
        cape = Parameter('cape', '228035', 1e-4),
        capeshear = Parameter('capeshear', '228036', 1e-4),
    )

    return params_db


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


def read_grib(fdb, req):
    fdb_reader = fdb.retrieve(req)
    eccodes_reader = eccodes.StreamReader(fdb_reader)
    messages = list(eccodes_reader)
    return messages


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

<<<<<<< Updated upstream
    # read reference file
    ref_reader = eccodes.FileReader(os.path.join(ref_dir, "ens_2t_avg_ref.grib"))
    avg_ref = [message.get_array("values") for message in ref_reader]
=======
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
>>>>>>> Stashed changes

    req_cf = {
        "class": "od",
        "expver": "0001",
        "stream": "enfo",
        "date": fc_date.strftime("%Y%m%d"),
        "time": fc_date.strftime("%H")+'00',
        "domain": "g",
        "type": "cf",
        "levtype": "sfc",
        "step": [6, 12, 18, 24],
        "param": "167",
    }

    req_pf = req_cf.copy()
    req_pf["type"] = "pf"

    avg = []

    fdb_dir = os.path.join(cfg.root_dir, 'fdb_data', ymdh)

    print("Control:")
    messages = eccodes.FileReader(os.path.join(fdb_dir, f'{cfg.param}_cf.grib'))
    print(messages)
    vals_cf = np.asarray([message.get_array('values') for message in messages])[:4]
    print(vals_cf.shape)
    avg_cf = compute_avg(vals_cf)
    
    avg.append(avg_cf)

    # read reference file
    ref_dir = os.path.join(cfg.root_dir, 'prepeps', ymdh)
    avg_ref = read_grib_file(os.path.join(ref_dir, f'eps_{cfg.param}{cfg.window_size:0>3}_{ymdh}_{cfg.window[0]:0>3}h_{cfg.window[1]:0>3}h.grib'))
    print(f'Reference array: {avg_ref.shape}')

    print("Perturbed:")
    avg_pf = []
    messages = eccodes.FileReader(os.path.join(fdb_dir, f'{cfg.param}_pf.grib'))
    print(messages)
    vals_pf = {}
    for member in range(1, 51):
        vals_pf[member] = []

    for message in messages:
        member = message.get('number')
        if int(message.get('stepRange')) <= 24:
            vals_pf[member].append(message.get_array('values'))

    for member in range(1, 51):
        vals_np = np.asarray(vals_pf[member])
        avg_pf = compute_avg(vals_np)
        avg.append(avg_pf)

    avg = np.asarray(avg)
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
        'step': '0-24',
        'quantile': ['{}:100'.format(i) for i in range(n_clim)],
        'param': '228004'
    }
    vals_clim = read_grib_file(os.path.join(cfg.clim_dir, f'clim_{cfg.param}{cfg.window_size:0>3}_{clim_ymd}_{cfg.window[0]:0>3}h_{cfg.window[1]:0>3}h_perc.grib'))

    return np.asarray(vals_clim)


def compute_efi(cfg, fc_avg, clim):

<<<<<<< Updated upstream
    ref_reader = eccodes.FileReader(os.path.join(ref_dir, 'efi', 'efi0_50_2t024_{}_012h_036h.grib'.format(fc_date.strftime("%Y%m%d%H"))))
    ref_efi = [message.get_array('values') for message in ref_reader]
=======
    efi = extreme.efi(clim, fc_avg, cfg.eps)
>>>>>>> Stashed changes

    write_grib(cfg.template, efi, cfg.out_dir, f'efi_{cfg.param}_{cfg.window_size:0>3}_{cfg.window[0]:0>3}h_{cfg.window[1]:0>3}h.grib')

    return efi


def compute_sot(cfg, fc_avg, clim):

    sot = {}
    for perc in [10, 90]:
        sot[perc] = extreme.sot(clim, fc_avg, perc, cfg.eps)
        write_grib(cfg.template, sot[perc], cfg.out_dir, f'sot{perc}_{cfg.param}_{cfg.window_size:0>3}_{cfg.window[0]:0>3}h_{cfg.window[1]:0>3}h.grib')

<<<<<<< Updated upstream
    ref_reader = eccodes.FileReader(os.path.join(ref_dir, 'sot', 'sot10_50_2t024_{}_012h_036h.grib'.format(fc_date.strftime("%Y%m%d%H"))))
    ref_sot10 = [message.get_array('values') for message in ref_reader]
    compare_arrs(sot[10], ref_sot10[0])

    ref_reader = eccodes.FileReader(os.path.join(ref_dir, 'sot', 'sot90_50_2t024_{}_012h_036h.grib'.format(fc_date.strftime("%Y%m%d%H"))))
    ref_sot90 = [message.get_array('values') for message in ref_reader]
    compare_arrs(sot[90], ref_sot90[0])
=======
>>>>>>> Stashed changes
    return sot


def check_results(cfg):

    check = 0

    fc_date = cfg.fc_date.strftime("%Y%m%d%H")

    print('Checking sot results')
    for perc in [10, 90]:
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
        self.param = args.param
        self.window = [int(i) for i in args.window.split('-')]
        self.window_size = self.window[1]-self.window[0]

        self.template = args.template
        self.eps = args.eps
        self.fdb = pyfdb.FDB()
        
        self.root_dir = args.root_dir
        self.ref_dir = os.path.join(self.root_dir, 'efi', self.fc_date.strftime("%Y%m%d%H"))
        self.out_dir = os.path.join(self.root_dir, 'efi_test', self.fc_date.strftime("%Y%m%d%H"))

        self.clim_date = climatology_date(self)
        self.clim_dir = os.path.join(self.root_dir, 'clim', self.clim_date.strftime("%Y%m%d"))

        print(f'Forecast date is {self.fc_date}')
        print(f'Climatology date is {self.clim_date}')
        print(f'Parameter is {self.param}')
        print(f'window is {self.window}, size {self.window_size}')
        print(f'Root directory is {self.root_dir}')
        print(f'Grib template is {self.template}')
        print(f'eps is {self.eps}')


def main(args=None):

    parser = argparse.ArgumentParser(description='Small python EFI test')
    parser.add_argument('fc_date', help='Forecast date')
    parser.add_argument('param', help='Parameter')
    parser.add_argument('window', help='Averaging window')
    parser.add_argument('root_dir', help='Root directory')
    parser.add_argument('template', help='GRIB template')
    parser.add_argument('--eps', type=float, help='epsilon factor')

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
