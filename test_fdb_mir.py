#!/usr/bin/env python3

from io import BytesIO

import eccodes
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

fdb = pyfdb.FDB()
reader = fdb.retrieve(req)

job = mir.Job(grid='1.0/1.0')
stream = BytesIO()
job.execute(reader, stream)

reader = eccodes.MemoryReader(stream.getvalue())
grib = next(reader)
grib.dump()
