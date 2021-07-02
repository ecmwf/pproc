
import numpy as np

import pyeccodes
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

ref_reader = pyeccodes.Reader("ens_2t_avg_ref.grib")
avg_ref = [message.get('values') for message in ref_reader]

print("Control:")
fdb_reader = fdb.retrieve(req_cf)
eccodes_reader = pyeccodes.Reader(fdb_reader)
vals_cf = np.asarray([message.get('values') for message in eccodes_reader])
avg_cf = compute_avg(vals_cf)
compare_arrs(avg_cf, avg_ref[0])

print("Perturbed:")
for member in range(1, 51):
    print(f"Member {member}:")
    req = req_pf.copy()
    req["number"] = member
    fdb_reader = fdb.retrieve(req)
    eccodes_reader = pyeccodes.Reader(fdb_reader)
    vals_pf = np.asarray([message.get('values') for message in eccodes_reader])
    avg_pf = compute_avg(vals_pf)
    compare_arrs(avg_pf, avg_ref[member])
