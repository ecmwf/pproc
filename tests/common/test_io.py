# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import eccodes
from functools import partial
import os
import pytest
import numpy as np

from pproc.common import dataset, io, parallel
from conftest import DATA_DIR


request = {
    "class": "od",
    "expver": "0001",
    "stream": "enfo",
    "date": "20240507",
    "domain": "g",
    "time": 12,
    "type": "cf",
    "levtype": "pl",
    "levelist": 250,
    "param": [138, 155],
    "step": range(0, 7, 3),
}


def test_fdb_read(fdb):
    data = io.fdb_read(fdb, request)
    assert len(data.coords["param"]) == 2
    assert len(data.coords["step"]) == 3
    assert isinstance(data.grib_template, eccodes.highlevel.message.GRIBMessage)
    assert not np.any(np.isnan(data))


def test_fdb_read_to_file(tmpdir, fdb):
    io.fdb_read_to_file(fdb, request, f"{tmpdir}/test.grib")
    data = [msg for msg in eccodes.FileReader(f"{tmpdir}/test.grib")]
    assert len(data) == 6


@pytest.mark.parametrize(
    "func",
    [
        io.fdb_read,
        io.fdb_read_to_file,
    ],
)
def test_fdb_no_data(tmpdir, fdb, func):
    no_data = request.copy()
    no_data["date"] = "20240508"
    if func == io.fdb_read_to_file:
        func = partial(func, file_out=f"{tmpdir}/test.grib")
    with pytest.raises(RuntimeError):
        func(fdb, no_data)


def test_fdb_missing_values(fdb):
    has_missing = request.copy()
    has_missing.pop("levelist")
    has_missing.update(
        {"stream": "waef", "param": 140232, "step": 12, "levtype": "sfc"}
    )
    data = io.fdb_read(fdb, has_missing)
    assert np.any(np.isnan(data))
    assert not np.any(data == 9999)


def test_fdb_target():
    target = io.target_from_location("fdb:")
    for msg in eccodes.FileReader(f"{DATA_DIR}/wind.grib"):
        msg.set("type", "em")
        target.write(msg)
    target.flush()
    req = request.copy()
    req["type"] = "em"
    data = list(eccodes.StreamReader(target.fdb.retrieve(req)))
    assert len(data) == 6


def test_file_target(tmpdir):
    target = io.target_from_location(f"file:{tmpdir}/test.grib")
    target.enable_recovery()
    messages = [
        msg
        for index, msg in enumerate(eccodes.FileReader(f"{DATA_DIR}/wind.grib"))
        if index < 5
    ]
    assert len(messages) == 5
    for msg in messages:
        target.write(msg)
    data = [msg for msg in eccodes.FileReader(f"{tmpdir}/test.grib")]
    assert len(data) == 5

    # Check files are overwritten
    for msg in messages:
        target.write(msg)
    data = [msg for msg in eccodes.FileReader(f"{tmpdir}/test.grib")]
    assert len(data) == 5


def test_fileset_target(tmpdir):
    target = io.target_from_location(f"fileset:{tmpdir}" + "/test_{step}.grib")
    target.enable_recovery()
    for msg in eccodes.FileReader(f"{DATA_DIR}/2t_ens.grib"):
        target.write(msg)
    files = os.listdir(tmpdir)
    assert set(x for x in files if x.endswith(".grib")) == set(
        f"test_{x}.grib" for x in range(12, 37, 6)
    )
    data = [msg for msg in eccodes.FileReader(f"{tmpdir}/test_12.grib")]
    assert len(data) == 6

    # Check files are overwritten
    for msg in eccodes.FileReader(f"{DATA_DIR}/2t_ens.grib"):
        target.write(msg)
    data = [msg for msg in eccodes.FileReader(f"{tmpdir}/test_12.grib")]
    assert len(data) == 6


def _write(target, message):
    # Modify parameter to distinguish from data already in FDB
    message.set("paramId", "228")
    target.write(message)
    target.flush()


@pytest.mark.parametrize(
    "loc, out_loc, reqs",
    [
        ["file:TMPDIR/test.grib", None, [{}]],
        [
            "fileset:TMPDIR/test_{step}.grib",
            "fileset:test",
            [{"step": x} for x in range(12, 37, 6)],
        ],
    ],
)
def test_target_parallel(tmpdir, fdb, loc, out_loc, reqs):
    loc = loc.replace("TMPDIR", str(tmpdir))
    if out_loc is None:
        out_loc = loc
    target = io.target_from_location(loc)
    target.enable_parallel()
    parallel.parallel_processing(
        _write, [(target, x) for x in eccodes.FileReader(f"{DATA_DIR}/2t_ens.grib")], 4
    )

    type_, path = loc.split(":")
    num_messages = 0
    for req in reqs:
        if type_ == "fileset":
            req["location"] = path
        reader = dataset.open_dataset({type_: {"test": req}}, out_loc)
        num_messages += len(list(reader))
    assert num_messages == 30
