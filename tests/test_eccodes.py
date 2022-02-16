# coding: utf-8
from io import BytesIO

import mir
import pyfdb

import eccodeshl

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

    ecc_reader = eccodeshl.MemoryReader(stream.getvalue())
    message = next(ecc_reader)

    val = message.get_array("values")

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
