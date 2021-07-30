# coding: utf-8
from io import BytesIO

import eccodes
import mir
import pyfdb

import eccodeshl
from eccodeshl.reader import codes_new_from_stream

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


def test_without_mir():
    fdb = pyfdb.FDB()
    result = fdb.retrieve(req)
    reader = eccodeshl.StreamReader(result)

    message = next(reader)

    val = message.get_array("values")

    return val


val = test_without_mir()
print(val)
