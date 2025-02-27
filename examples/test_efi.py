import argparse
import time
import numpy as np
import eccodes
import pyfdb
from earthkit.meteo import extreme
import xarray as xr


def fdb_read_fc(fdb, fc_date):

    values = []

    cf_req = {
        'class': 'od',
        'expver': '0075',
        'stream':'enfo',
        'date': fc_date,
        'time': '0000',
        'domain': 'g',
        'type': 'cf',
        'levtype': 'sfc',
        'step': '6',
        'param': '167'
    }
    pf_req = cf_req.copy()
    pf_req['type'] = 'pf'
    pf_req['number'] = list(range(1, 51))

    for req in [cf_req, pf_req]:
        print(req)
        fdb_reader = fdb.retrieve(req)

        eccodes_reader = eccodes.StreamReader(fdb_reader)
        for message in eccodes_reader:
            val = message.get('values')
            print(val.shape)
            values.append(val)

    return np.asarray(values)

def fdb_read_fc_eccodes(fdb, fc_date):

    values = []

    cf_req = {
        'class': 'od',
        'expver': '0075',
        'stream':'enfo',
        'date': fc_date,
        'time': '0000',
        'domain': 'g',
        'type': 'cf',
        'levtype': 'sfc',
        'step': '6',
        'param': '167'
    }
    pf_req = cf_req.copy()
    pf_req['type'] = 'pf'
    pf_req['number'] = list(range(1, 51))

    for req in [cf_req, pf_req]:
        print(req)
        fdb_reader = fdb.retrieve(req)

        while 1:
            igrib = eccodes.codes_new_from_message(fdb_reader)
            if igrib is None: break

            val = eccodes.codes_get_values()
            values.append(val)

            eccodes.codes_release(igrib)

    return np.asarray(values)

def fdb_read_clim(fdb, clim_date):

    values = []

    req = {
        'class': 'od',
        'expver': '0075',
        'stream': 'efhs',
        'date': clim_date,
        'time': '0000',
        'domain': 'g',
        'type': 'cd',
        'levtype': 'sfc',
        'step': '0-24',
        'quantile': ['{}:100'.format(i) for i in range(101)],
        'param': '228004'
    }

    fdb_reader = fdb.retrieve(req)

    eccodes_reader = eccodes.StreamReader(fdb_reader)
    for message in eccodes_reader:
        val = message.get('values')
        print(val.shape)
        values.append(val)

    return np.asarray(values)

    # values = []

    # clim_file = 'efi_sot/clim/clim_2t024_{}_000h_024h_perc.grib'.format(clim_date)
    # da = xr.open_dataarray(clim_file, engine='cfgrib')
    # print(da)
    # return da.values


def read_efi(efi_file):
    da = xr.open_dataarray(efi_file, engine='cfgrib')
    return da.values


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Small python EFI test')
    parser.add_argument('fc_date', help='Forecast date')
    parser.add_argument('clim_date', help='climatology date')
    parser.add_argument('efi', help='EFI file')

    args = parser.parse_args()
    clim_date = args.clim_date
    fc_date = args.fc_date
    efi_file = args.efi

    fdb = pyfdb.FDB()

    fc = fdb_read_fc_eccodes(fdb, fc_date)
    print(fc.shape)
    clim = fdb_read_clim(fdb, clim_date)
    print(clim.shape)

    print(fc[:, 0])
    print(clim[:, 0])
    t1 = time.perf_counter()
    efi_array = extreme.efi(clim.astype(np.float64), fc)
    print('time to compute efi: {}'.format(time.perf_counter() - t1))
    print(efi_array[0])
    
    efi_check = read_efi(efi_file)

    if not np.allclose(efi_array, efi_check):
        mask = np.logical_not(np.isclose(efi_array, efi_check))
        print(efi_array[mask])
        print(efi_check[mask])
