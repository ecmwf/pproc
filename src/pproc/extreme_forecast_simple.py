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

import eccodes
from earthkit.meteo import extreme


def compute_efi(fcs, clim, eps):

    efi = extreme.efi(clim, fcs, eps)

    return efi


def compute_sot(fcs, clim, sot_values, eps):

    sot = {}
    for perc in sot_values:
        sot[perc] = extreme.sot(clim, fcs, perc, eps=eps)

    return sot


def read_grib(in_file):
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


def main(args=None):

    parser = argparse.ArgumentParser(description='Small python EFI test')
    parser.add_argument('fc_file', help='Forecast input file')
    parser.add_argument('clim_file', help='Climatology input file')
    parser.add_argument('template', help='GRIB template')
    parser.add_argument('out_dir', help='Output directory')
    parser.add_argument('--sot', nargs="+", type=int, help='SOT percentiles values')
    parser.add_argument('--eps', type=float, help='epsilon factor')
    
    args = parser.parse_args(args)

    sot_num = args.sot
    if not isinstance(sot_num, list):
        sot_num = [sot_num]
    
    fcs = read_grib(args.fc_file)
    clim = read_grib(args.clim_file)
    print(np.mean(fcs))
    print(np.mean(clim))
    
    fc_name = os.path.basename(args.fc_file).split('_')[0]
    
    efi = compute_efi(fcs, clim, eps=args.eps)
    print(np.mean(efi))
    
    efifile = 'efi' + os.path.basename(args.fc_file)[3:]
    write_grib(args.template, efi, args.out_dir, efifile)
    
    if fc_name == 'eps':
        sot = compute_sot(fcs, clim, sot_num, eps=args.eps)
        for val in sot_num:
            sotfile = 'sot' + str(val) + os.path.basename(args.fc_file)[3:]
            write_grib(args.template, sot[val], args.out_dir, sotfile)
    

if __name__ == "__main__":
    main()
