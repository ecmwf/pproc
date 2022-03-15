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

import eccodeshl
from meteokit import extreme


def compute_efi(fcs, clim, eps):

    efi = extreme.efi(clim, fcs, eps)

    return efi


def compute_sot(fcs, clim, sot_values eps, missing):

    sot = {}
    for perc in sot_values:
        sot[perc] = extreme.sot(clim, fcs, perc, eps=eps, missing=missing)

    return sot


def read_grib(in_file):
    reader = eccodeshl.FileReader(in_file)
    data = [message.get_array("values") for message in reader]
    #print(data)

    return np.asarray(data)


def write_grib(template, data, out_dir, out_name, missing=None):
    reader = eccodeshl.FileReader(template)
    messages = list(reader)
    message = messages[0]

    if missing is not None:
        message.set('missingValue', missing)

    message.set_array('values', data)

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
    missing = -99999

    sot_num = args.sot
    if not isinstance(sot_num, list):
        sot_num = [sot_num]
    
    fcs = read_grib(args.fc_file)
    clim = read_grib(args.clim_file)
    
    fc_name = os.path.basename(args.fc_file).split('_')[0]
    
    efi = compute_efi(fcs, clim, eps=args.eps)
    
    efifile = 'efi' + os.path.basename(args.fc_file)[3:]
    write_grib(args.template, efi, args.out_dir, efifile)
    
    if fc_name == 'eps':
        sot = compute_sot(fcs, clim, sot_num eps=args.eps, missing=missing)
        for val in sot_num:
            sotfile = 'sot' + str(val) + os.path.basename(args.fc_file)[3:]
            write_grib(args.template, sot[val], args.out_dir, sotfile, missing=missing)
    

if __name__ == "__main__":
    main()
