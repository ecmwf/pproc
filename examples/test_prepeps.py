
import numpy as np

import eccodes
import pyfdb

def compute_avg(fields):
    nsteps = fields.shape[0]
    return np.sum(fields, axis=0) / nsteps

def compare_arrs(act, exp):
    if np.allclose(act, exp):
        print("OK")
    else:
        mask = np.logical_not(np.isclose(act, exp))
        print("{}/{} values differ, max rel diff {}".format(
            np.sum(mask), act.size, (np.abs(act - exp) / exp).max()))


req_cf = {
    "class": "od",
    "expver": "0075",
    "stream": "enfo",
    "date": "20210628",
    "time": "0000",
    "domain": "g",
    "type": "cf",
    "levtype": "sfc",
    "step": [6, 12, 18, 24],
    "param": "167"
}

req_pf = req_cf.copy()
req_pf["type"] = "pf"

fdb = pyfdb.FDB()

ref_reader = eccodes.FileReader("ens_2t_avg_ref.grib")
avg_ref = [message.get_array('values') for message in ref_reader]

with open("ens_2t_avg.grib", "wb") as outfile:
    print("Control:")
    fdb_reader = fdb.retrieve(req_cf)
    eccodes_reader = eccodes.StreamReader(fdb_reader)
    messages = list(eccodes_reader)
    vals_cf = np.asarray([message.get_array('values') for message in messages])
    avg_cf = compute_avg(vals_cf)
    compare_arrs(avg_cf, avg_ref[0])

    message = messages[0]
    message.set_array('values', avg_cf)
    # FIXME: set metadata
    message.write_to(outfile)

    print("Perturbed:")
    for member in range(1, 51):
        print(f"Member {member}:")
        req = req_pf.copy()
        req["number"] = member
        fdb_reader = fdb.retrieve(req)
        eccodes_reader = eccodes.StreamReader(fdb_reader)
        messages = list(eccodes_reader)
        vals_pf = np.asarray([message.get_array('values') for message in messages])
        avg_pf = compute_avg(vals_pf)
        compare_arrs(avg_pf, avg_ref[member])

        message = messages[0]
        message.set_array('values', avg_pf)
        # FIXME: set metadata
        message.write_to(outfile)
