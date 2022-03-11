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


import argparse
import numpy as np

import eccodeshl
from meteokit import extreme


def compute_efi(fcs, clim):

    efi = extreme.efi(clim, fcs)

    return efi


def compute_sot(fcs, clim, sot_values):

    sot = {}
    for perc in sot_values:
        sot[perc] = extreme.sot(clim, fcs, perc)

    return sot


def read_grib(in_file):
    reader = eccodeshl.FileReader(in_file)
    data = [message.get_array("values") for message in reader]
    print(data)

    return data


def write_grib(template, data, out_dir, out_name):
    reader = eccodeshl.FileReader(template)
    messages = list(reader)
    message = messages[0]
    message.set_array('values', data)

    out_file = os.path.join(out_dir, out_name)
    message.write_to(out_file)


def main(args=None):

    parser = argparse.ArgumentParser(description='Small python EFI test')
    parser.add_argument('fc_file', help='Forecast input file')
    parser.add_argument('clim_file', help='Climatology input file')
    parser.add_argument('template', help='GRIB template')
    parser.add_argument('out_dir', help='Output directory')
    parser.add_argument('--sot', nargs="+", type=int, help='SOT percentiles values')

    args = parser.parse_args(args)

    sot_values = args.sot
    if not isinstance(sot, list):
        sot_values = [sot_values]
    
    fcs = read_grib(args.fc_file)
    clim = read_grib(args.clim_file)

    efi = compute_efi(fcs, clim)

    sot = compute_sot(fcs, clim, sot_values)

    write_grib(args.template, efi, args.out_dir, 'efi.grib')
    for val in sot_values:
        write_grib(args.template, sot[0], args.out_dir, f'sot_{val}.grib')

if __name__ == "__main__":
    main()
