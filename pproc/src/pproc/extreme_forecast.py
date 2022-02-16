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

# import eccodes
import pyfdb

import eccodeshl
from meteokit import extreme


def compute_avg(fields):
    nsteps = fields.shape[0]
    return np.sum(fields, axis=0) / nsteps

def compare_arrs(dev, ref):
    if np.allclose(dev, ref, rtol=1e-4):
        print("OK")
    else:
        mask = np.logical_not(np.isclose(dev, ref, rtol=1e-4))
        print(dev[mask], ref[mask])
        print("{}/{} values differ, max rel diff {}".format(
            np.sum(mask), dev.size, (np.abs(dev - ref) / ref).max()))


def read_grib(fdb, req):
    fdb_reader = fdb.retrieve(req)
    eccodes_reader = eccodeshl.StreamReader(fdb_reader)
    messages = list(eccodes_reader)
    return messages


def write_avg(template_message, avg, filename):

    with open(filename, "ab") as outfile:
        message = template_message
        message.set_array("values", avg)
        # FIXME: set metadata
        message.write_to(outfile)


def compute_forecast_average(fdb, fc_date, ref_dir, out_dir):

    # read reference file
    ref_reader = eccodeshl.FileReader(os.path.join(ref_dir, "ens_2t_avg_ref.grib"))
    avg_ref = [message.get_array("values") for message in ref_reader]

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

    print("Control:")
    messages = read_grib(fdb, req_cf)
    print(messages)
    vals_cf = np.asarray([message.get_array('values') for message in messages])
    avg_cf = compute_avg(vals_cf)
    compare_arrs(avg_cf, avg_ref[0])
    cf_avg_file = 'cf_avg.grib'
    write_avg(messages[0], avg_cf, cf_avg_file)
    avg.append(avg_cf)

    print("Perturbed:")
    avg_pf = []
    for member in range(1, 51):
        print("Member {}:".format(member))
        req = req_pf.copy()
        req["number"] = member
        messages = read_grib(fdb, req)
        print(messages)
        vals_pf = np.asarray([message.get_array('values') for message in messages])
        avg_pf = compute_avg(vals_pf)
        compare_arrs(avg_pf, avg_ref[member])

        pf_avg_file = 'pf_avg.grib'
        write_avg(messages[0], avg_pf, pf_avg_file)
        avg.append(avg_pf)

    return np.asarray(avg)


def read_clim(fdb, clim_date, n_clim=101):

    req = {
        'class': 'od',
        'expver': '0001',
        'stream': 'efhs',
        'date': clim_date.strftime("%Y%m%d"),
        'time': '0000',
        'domain': 'g',
        'type': 'cd',
        'levtype': 'sfc',
        'step': '0-24',
        'quantile': ['{}:100'.format(i) for i in range(n_clim)],
        'param': '228004'
    }
    messages = read_grib(fdb, req)
    print('Climatology:')
    print(messages)
    vals_clim = []
    for message in messages:
        q = message.get('quantile')
        print(q)
        vals_clim.append(message.get_array('values'))

    return np.asarray(vals_clim)


def compute_efi(fdb, fc_date, fc_avg, clim, ref_dir, out_dir):

    efi = extreme.efi(clim, fc_avg)

    ref_reader = eccodeshl.FileReader(os.path.join(ref_dir, 'efi', 'efi0_50_2t024_{}_012h_036h.grib'.format(fc_date.strftime("%Y%m%d%H"))))
    ref_efi = [message.get_array('values') for message in ref_reader]

    compare_arrs(efi, ref_efi[0])

    return efi


def compute_sot(fdb, fc_date, fc_avg, clim, ref_dir, out_dir):

    sot = {}
    for perc in [10, 90]:
        sot[perc] = extreme.sot(clim, fc_avg, perc)

    ref_reader = eccodeshl.FileReader(os.path.join(ref_dir, 'sot', 'sot10_50_2t024_{}_012h_036h.grib'.format(fc_date.strftime("%Y%m%d%H"))))
    ref_sot10 = [message.get_array('values') for message in ref_reader]
    compare_arrs(sot[10], ref_sot10[0])

    ref_reader = eccodeshl.FileReader(os.path.join(ref_dir, 'sot', 'sot90_50_2t024_{}_012h_036h.grib'.format(fc_date.strftime("%Y%m%d%H"))))
    ref_sot90 = [message.get_array('values') for message in ref_reader]
    compare_arrs(sot[90], ref_sot90[0])
    return sot


def climatology_date(fc_date):

    weekday = fc_date.weekday()

    # friday to monday -> take previous monday clim, else previous thursday clim
    if weekday == 0 or weekday > 3:
        clim_date = fc_date - timedelta(days=(weekday+4)%7)
    else:
        clim_date = fc_date - timedelta(days=weekday)

    return clim_date


def main(args=None):

    parser = argparse.ArgumentParser(description='Small python EFI test')
    parser.add_argument('fc_date', help='Forecast date')
    parser.add_argument('ref_dir', help='Reference directory')

    args = parser.parse_args(args)
    fc_date = datetime.strptime(args.fc_date,"%Y%m%d%H")
    clim_date = climatology_date(fc_date)
    print(fc_date)
    print(clim_date)
    ref_dir = os.path.join(args.ref_dir, args.fc_date)
    out_dir = None#args.out_dir

    fdb = pyfdb.FDB()

    fc_avg = compute_forecast_average(fdb, fc_date, ref_dir, out_dir)

    clim = read_clim(fdb, clim_date)
    print(clim)

    efi = compute_efi(fdb, fc_date, fc_avg, clim, ref_dir, out_dir)

    sot = compute_sot(fdb, fc_date, fc_avg, clim, ref_dir, out_dir)


if __name__ == "__main__":
    main()
