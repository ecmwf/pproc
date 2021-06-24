import argparse
import time
import numpy as np
import pyeccodes
import pyfdb
import efi
import xarray as xr


def get_ens(fc_date):

    fdb = pyfdb.FDB()
    values = []

    for i in range(51):
        if i == 0:
            ens_type = 'cf'
        else:
            ens_type = 'pf'
        req = {
            'class': 'od',
            'expver': '0075',
            'stream':'enfo',
            'date': fc_date,
            'time': '0000',
            'domain': 'g',
            'type': ens_type,
            'levtype': 'sfc',
            'step': '6',
            'param': '167'
        }
        if i > 0 :
            req['number'] = str(i)

        print(i)
        fdb_reader = fdb.retrieve(req)

        eccodes_reader = pyeccodes.Reader(fdb_reader)
        message = next(eccodes_reader)
        values.append(message.get('values'))

    return np.asarray(values)

# {class=od,expver=0075,stream=efhs,date=20210621,time=0000,domain=g}{type=cd,levtype=sfc}{step=0-24,quantile=9:100,param=228004}
def read_clim(clim_file):
    da = xr.open_dataarray(clim_file, engine='cfgrib')
    print(da)
    return da.values


def read_efi(efi_file):
    da = xr.open_dataarray(efi_file, engine='cfgrib')
    return da.values


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Small python EFI test')
    parser.add_argument('clim', help='climatology file')
    parser.add_argument('efi', help='EFI file')
    parser.add_argument('date', help='Forecast date')

    args = parser.parse_args()
    clim_file = args.clim
    fc_date = args.date
    efi_file = args.efi

    ens = get_ens(fc_date)
    print(ens.shape)
    clim = read_clim(clim_file)
    print(clim.shape)

    print(ens[:, 0])
    print(clim[:, 0])
    t1 = time.perf_counter()
    efi_array = efi.efi(clim.astype(np.float64), ens)
    print('time to compute efi: {}'.format(time.perf_counter() - t1))
    
    efi_check = read_efi(efi_file)

    if not np.allclose(efi_array, efi_check):
        mask = np.logical_not(np.isclose(efi_array, efi_check))
        print(efi_array[mask])
        print(efi_check[mask])
