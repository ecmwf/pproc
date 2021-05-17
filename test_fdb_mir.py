
from io import BytesIO

import mir
import pyeccodes
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

job = mir.MIRJob().set('grid', '1.0/1.0')
stream = BytesIO()
job.execute(mir.GribPyIOInput(reader), mir.GribPyIOOutput(stream))

stream.seek(0)
stream.mode = 'rb'  # for pyeccodes
reader = pyeccodes.Reader(stream)
grib = next(reader)
grib.dump()
