
import eccodes
import gribapi
from gribapi import ffi

from .message import Message

@ffi.callback("long(*)(void*, void*, long)")
def pyread_callback(payload, buf, length):
    stream = ffi.from_handle(payload)
    read = stream.read(length)
    n = len(read)
    ffi.buffer(buf, length)[:n] = read
    return n if n > 0 else -1  # -1 means EOF


cstd = ffi.dlopen(None)
ffi.cdef("void free(void* pointer);")
ffi.cdef("void* wmo_read_any_from_stream_malloc(void* stream_data, long (*stream_proc)(void*, void* buffer, long len), size_t* size, int* err);")
def codes_new_from_stream(stream):
    sh = ffi.new_handle(stream)
    length = ffi.new("size_t*")
    err = ffi.new("int*")
    err, buf = gribapi.err_last(gribapi.lib.wmo_read_any_from_stream_malloc)(sh, pyread_callback, length)
    buf = ffi.gc(buf, cstd.free, size=length[0])
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


class Reader:
    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        return self

    def __next__(self):
        handle = codes_new_from_stream(self.stream)
        if handle is None:
            raise StopIteration
        return Message(handle)
