# coding: utf-8
from io import BytesIO

import eccodes
import gribapi
from gribapi import ffi
import mir
import pyfdb

req = {
    "class": "rd",
    "expver": "xxxx",
    "stream": "oper",
    "date": "20191110",
    "time": "0000",
    "domain": "g",
    "type": "an",
    "levtype": "pl",
    "step": "0",
    "levelist": "400",
    "param": "138"
}


def test_with_mir():
    fdb = pyfdb.FDB()
    reader = fdb.retrieve(req)

    job = mir.Job(grid="1/1")
    stream = BytesIO()
    job.execute(reader, stream)

    handle = eccodes.codes_new_from_message(stream.getvalue())

    val = eccodes.codes_get_array(handle, "values")

    eccodes.codes_release(handle)

    return val


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


def test_without_mir():
    fdb = pyfdb.FDB()
    reader = fdb.retrieve(req)

    handle = codes_new_from_stream(reader)

    val = eccodes.codes_get_array(handle, "values")

    eccodes.codes_release(handle)

    return val


val = test_without_mir()
