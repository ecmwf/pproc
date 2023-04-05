#!/usr/bin/env python3

from io import BytesIO

import eccodes
import mir
import pyfdb
from pproc import common

req = {
    "class": "od",
    "expver": "0001",
    "stream": "oper",
    "date": "20221212",
    "time": "0000",
    "domain": "g",
    "type": "fc",
    "levtype": "sfc",
    "step": [0,1,2],
    "param": "167"
}

fdb = pyfdb.FDB()

a = common.fdb_read(fdb, req, mir_options={'grid': '1.0/1.0'})
print(a)

# fdb_reader = fdb.retrieve(req)

# job = mir.Job(grid='1.0/1.0')
# stream = BytesIO()
# job.execute(fdb_reader, stream)
# stream.seek(0)
# fdb_reader = stream

# ecreader = eccodes.StreamReader(fdb_reader)
# for message in ecreader:
#    print(message.get('step'))
#    print(message.get_size('values'))
