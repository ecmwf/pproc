
import argparse
import time
import numpy as np
import xarray as xr

from earthkit.meteo import extreme
from io import BytesIO

import eccodes
import gribapi
from gribapi import ffi
import mir
import pyfdb

req = {
    'class': 'od',
    'expver': '0001',
    'stream':'enfo',
    'date': {0},
    'time': '12',
    'domain': 'g',
    'type': 'cf',
    'levtype': 'sfc',
    'step': '18/to/036/by/6',
    'param': '167.128'
}


@ffi.callback("long(*)(void*, void*, long)")
def pyread_callback(payload, buf, length):
    stream = ffi.from_handle(payload)
    read = stream.read(length)
    n = len(read)
    ffi.buffer(buf, length)[:n] = read
    return n if n > 0 else -1  # -1 means EOF


ffi.cdef("void* wmo_read_any_from_stream_malloc(void* stream_data, long (*stream_proc)(void*, void* buffer, long len), size_t* size, int* err);")
def codes_new_from_stream(stream):
    sh = ffi.new_handle(stream)
    length = ffi.new("size_t*")
    err = ffi.new("int*")
    err, buf = gribapi.err_last(gribapi.lib.wmo_read_any_from_stream_malloc)(sh, pyread_callback, length)
    # TODO: release buf, maybe ffi.gc()?
    if err:
        if err != gribapi.lib.GRIB_END_OF_FILE:
            gribapi.GRIB_CHECK(err)
        return None

    # TODO: remove the extra copy?
    handle = gribapi.lib.grib_handle_new_from_message_copy(ffi.NULL, buf, length[0])
    if handle == ffi.NULL:
        return None
    else:
        return gribapi.put_handle(handle)


def read_fc(date):
    fdb = pyfdb.FDB()
    reader = fdb.retrieve(req.format(date))

    handle = codes_new_from_stream(reader)

    val = eccodes.codes_get_array(handle, "values")

    eccodes.codes_release(handle)

    return val


# def read_fc(date):
#     cf_file = 'epscontrol_2t024_{}_012h_036h.grib'.format(date)
#     da_cf = xr.open_dataarray(cf_file, engine='cfgrib')
#     pf_file = 'epsensemble_2t024_{}_012h_036h.grib'.format(date)
#     da_pf = xr.open_dataarray(pf_file, engine='cfgrib')
#     print('FORECAST DATASET:')
#     print(da_pf)
#     return np.concatenate([[da_cf.values], da_pf.values], axis=0)


def read_clim(date):
    clim_file = 'clim_2t024_{}_000h_024h_perc.grib'.format(date)
    da = xr.open_dataarray(clim_file, engine='cfgrib')
    print('CLIMATOLOGY DATASET:')
    print(da)
    return da.values


def read_sot(date, perc):
    sot_file = 'sot{}_50_2t024_{}_012h_036h.grib'.format(perc, date)
    da = xr.open_dataarray(sot_file, engine='cfgrib')
    # print('SOT DATASET:')
    # print(da)
    return da.values


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Small python SOT test')
    parser.add_argument('fc_date', help='Forecast date')
    parser.add_argument('clim_date', help='climatology date')

    args = parser.parse_args()
    clim_date = args.clim_date
    fc_date = args.fc_date

    fc = read_fc(fc_date)
    print(fc.shape)
    clim = read_clim(clim_date)
    print(clim.shape)

    print(fc[0])
    print(clim[:, 0])
    for perc in [10, 90]:
        t1 = time.perf_counter()
        sot_array = extreme.sot(clim, fc, perc)
        print('time to compute sot: {}'.format(time.perf_counter() - t1))
        print(sot_array[0])
        
        sot_check = read_sot(fc_date, perc)
        print(sot_check[0])

        if not np.allclose(sot_array, sot_check):
            mask = np.logical_not(np.isclose(sot_array, sot_check))
            print(sot_array[mask].shape)
            print(sot_array[mask])
            print(sot_check[mask])

#[-1.93559592 -2.04790934 -2.02790152 ... -6.23949824 -6.22566589 -6.09865446]
#[-1.9355959 -2.0479093 -2.0279014 ... -6.239498  -6.225666  -6.0986543]
