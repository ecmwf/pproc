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

# import eccodes
import pyfdb

import eccodeshl


def compute_avg(fields):
    nsteps = fields.shape[0]
    return np.sum(fields, axis=0) / nsteps


def compare_arrs(act, exp):
    if np.allclose(act, exp):
        print("OK")
    else:
        mask = np.logical_not(np.isclose(act, exp))
        print(
            "{}/{} values differ, max rel diff {}".format(
                np.sum(mask), act.size, (np.abs(act - exp) / exp).max()
            )
        )


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
        "expver": "0075",
        "stream": "enfo",
        "date": fc_date,
        "time": "0000",
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
    vals_cf = np.asarray([message.get_array("values") for message in messages])
    avg_cf = compute_avg(vals_cf)
    compare_arrs(avg_cf, avg_ref[0])
    write_avg(messages[0], avg_cf, cf_avg_file)
    avg.append(avg_cf)

    print("Perturbed:")
    avg_pf = []
    for member in range(1, 51):
        print(f"Member {member}:")

        messages = read_grib(fdb, req)
        vals_pf = np.asarray([message.get_array("values") for message in messages])
        avg_pf = compute_avg(vals_pf)
        compare_arrs(avg_pf, avg_ref[member])

        req = req_pf.copy()
        req["number"] = member
        write_avg(messages[0], avg_ref, pf_avg_file)
        avg.append(avg_pf)

    return avg


def read_clim(fdb, clim_date, n_clim=101):

    req = {
        "class": "od",
        "expver": "0075",
        "stream": "efhs",
        "date": clim_date,
        "time": "0000",
        "domain": "g",
        "type": "cd",
        "levtype": "sfc",
        "step": "0-24",
        "quantile": ["{}:100".format(i) for i in range(n_clim)],
        "param": "228004",
    }
    messages = read_grib(fdb, req)
    vals_clim = np.asarray([message.get_array("values") for message in messages])

    return vals_clim


def compute_efi(fdb, fc_date, fc_avg, clim, ref_dir, out_dir):

    efi = extreme.efi(clim, fc_avg)

    return efi


def compute_sot(fdb, fc_date, fc_avg, clim, ref_dir, out_dir):

    sot = {}
    for perc in [10, 90]:
        sot[perc] = extreme.sot(clim, fc, perc)

    return sot


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Small python EFI test")
    parser.add_argument("fc_date", help="Forecast date")
    parser.add_argument("clim_date", help="climatology date")
    parser.add_argument("ref_dir", help="Reference directory")
    parser.add_argument("out_dir", help="Output directory")

    fc_date = args.fc_date
    clim_date = args.clim_date
    ref_dir = args.ref_dir
    out_dir = args.out_dir

    fdb = pyfdb.FDB()

    fc_avg = compute_forecast_average(fdb, fc_date, ref_dir, out_dir)

    clim = read_clim(fdb, clim_date)

    efi = compute_efi(fdb, fc_date, fc_avg, clim, ref_dir, out_dir)

    sot = compute_sot(fdb, fc_date, fc_avg, clim, ref_dir, out_dir)
