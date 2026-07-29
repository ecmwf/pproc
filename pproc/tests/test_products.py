# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import pytest

import pyfdb
import eccodes

from pproc.probabilities import main as prob_main
from pproc.ensms import main as ensms_main
from pproc.extreme import main as extreme_main
from pproc.quantiles import main as quantiles_main
from pproc.thermal_indices import main as thermo_main
from pproc.clustereps.__main__ import main as clustereps_main
from ppcore.product import main as ppcore_main
from conftest import DATA_DIR

TEST_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize(
    "product, main, custom_args, req, length",
    [
        [
            "prob",
            prob_main,
            [],
            {"type": "ep", "param": 131073, "step": ["12", "12-36"]},
            2,
        ],
        [
            "t850",
            prob_main,
            [],
            {
                "levtype": "pl",
                "levelist": 850,
                "type": "ep",
                "param": [131022, 133093],
                "step": [0, 12],
            },
            4,
        ],
        [
            "ensms",
            ensms_main,
            [],
            {"type": "em", "param": 167, "step": [12, 36]},
            2,
        ],
        [
            "extreme",
            extreme_main,
            [],
            {
                "type": "efi",
                "param": 167,
                "step": "12-36",
            },
            1,
        ],
        [
            "quantiles",
            quantiles_main,
            [],
            {
                "type": "pb",
                "param": 167,
                "step": [12, 18, 24, 30, 36],
                "quantile": ["{}:3".format(x) for x in range(4)],
            },
            20,
        ],
        [
            "wind",
            ensms_main,
            [],
            {
                "type": "es",
                "levtype": "pl",
                "levelist": [250, 850],
                "param": 10,
                "step": [0, 3, 6],
            },
            6,
        ],
        [
            "thermo",
            thermo_main,
            [],
            {
                "type": "fc",
                "stream": "oper",
                "date": 20240605,
                "time": 00,
                "param": [
                    261002,
                    261001,
                    260004,
                    260005,
                    260255,
                    261016,
                    261018,
                    261015,
                    261023,
                    261014,
                    260242,
                ],
                "step": list(range(0, 4)),
            },
            34,
        ],
        [
            "clustereps",
            clustereps_main,
            [
                "--set",
                "output_root={test_dir}",
                "--set",
                "clim_dir={DATA_DIR}/clustclim",
                "--set",
                "ncomp_file={test_dir}/NEOF",
                "--set",
                "outputs.cen_anomalies.target=file:{test_dir}/clm_anom.grib",
                "--set",
                "outputs.rep_anomalies.target=file:{test_dir}/clr_anom.grib",
            ],
            {
                "levtype": "pl",
                "levelist": 500,
                "domain": "h",
                "type": "cm",
                "param": 129,
                "step": [72, 84, 96],
                "number": range(1, 7),
            },
            18,
        ],
    ],
    ids=[
        "prob",
        "t850",
        "ensms",
        "extreme",
        "quantiles",
        "wind",
        "thermofeel",
        "clustereps",
    ],
)
def test_products_v2(tmpdir, monkeypatch, fdb, product, main, custom_args, req, length):
    monkeypatch.chdir(tmpdir)  # To avoid polluting cwd with grib templates
    args = [product, "--config", f"{TEST_DIR}/templates/v2/{product}.yaml"] + [
        x.format_map(
            {
                "test_dir": str(tmpdir),
                "TEST_DIR": str(TEST_DIR),
                "DATA_DIR": str(DATA_DIR),
            }
        )
        for x in custom_args
    ]
    monkeypatch.setattr("sys.argv", args)
    main()
    test_fdb = pyfdb.FDB()
    request = {
        "class": "od",
        "expver": "0001",
        "stream": "enfo",
        "date": 20240507,
        "time": 12,
        "levtype": "sfc",
        "domain": "g",
    }
    request.update(req)
    messages = list(eccodes.StreamReader(test_fdb.retrieve(request)))
    assert len(messages) == length


@pytest.mark.parametrize(
    "product, req, length",
    [
        [
            "ensms",
            {"type": "em", "param": 167, "step": [12, 36]},
            2,
        ],
        [
            "prob",
            {"type": "ep", "param": 131073, "step": ["12", "12-36"]},
            2,
        ],
        [
            "quantiles",
            {
                "type": "pb",
                "param": 167,
                "step": [12, 18, 24, 30, 36],
                "quantile": ["{}:3".format(x) for x in range(4)],
            },
            20,
        ],
        [
            "thermo",
            {
                "stream": "oper",
                "type": "fc",
                "param": [261002, 261001, 260004, 260005, 260255, 260242, 261016, 261018, 261015, 261014, 261023],
                "step": [1],
                "date": 20260720,
                "time": 12,
            },
            11,
        ],
    ],
    ids=["ensms", "prob", "quantiles", "thermo"],
)
def test_products_v3(tmpdir, monkeypatch, fdb, product, req, length):
    monkeypatch.chdir(tmpdir)  # To avoid polluting cwd with grib templates
    args = [product, "--config", f"{TEST_DIR}/templates/v3/{product}.yaml"]
    monkeypatch.setattr("sys.argv", args)
    ppcore_main()
    test_fdb = pyfdb.FDB()
    request = {
        "class": "od",
        "expver": "0001",
        "stream": "enfo",
        "date": 20240507,
        "time": 12,
        "levtype": "sfc",
        "domain": "g",
    }
    request.update(req)
    messages = list(eccodes.StreamReader(test_fdb.retrieve(request)))
    assert len(messages) == length
