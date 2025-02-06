import os
import pytest
import shutil
import yaml

import pyfdb
import eccodes

from pproc.probabilities import main as prob_main
from pproc.anomaly_probs import main as anomaly_prob_main
from pproc.ensms import main as ensms_main
from pproc.extreme import main as extreme_main
from pproc.quantiles import main as quantiles_main
from pproc.wind import main as wind_main
from pproc.thermal_indices import main as thermo_main
from pproc.clustereps.__main__ import main as clustereps_main
from conftest import DATA_DIR

TEST_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize(
    "product, main, custom_args, pass_args, req, length",
    [
        [
            "prob",
            prob_main,
            ["-d", "2024050712", "--in-ens", "fdb:ens", "--out-prob", "fdb:"],
            False,
            {"type": "ep", "param": 131073, "step": ["12", "12-36"]},
            2,
        ],
        [
            "t850",
            anomaly_prob_main,
            [
                "-d",
                "2024050712",
                "--in-ens",
                "fdb:ens",
                "--in-clim",
                "fdb:clim",
                "--out-prob",
                "fdb:",
            ],
            False,
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
            [
                "--in-ens",
                "fdb:ens",
                "--out-mean",
                "fdb:",
                "--out-std",
                "fdb:",
            ],
            False,
            {"type": "em", "param": 167, "step": [12, 36]},
            2,
        ],
        [
            "extreme",
            extreme_main,
            [
                "--in-ens",
                "fdb:ens",
                "--in-clim",
                "fdb:clim",
                "--out-efi",
                "fdb:",
                "--out-sot",
                "fdb:",
            ],
            False,
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
            [
                "--in-ens",
                "fdb:ens",
                "--out-quantiles",
                "fdb:",
            ],
            True,
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
            wind_main,
            [
                "--in-ens",
                "fdb:ens",
                "--in-det",
                "fdb:det",
                "--out-eps-mean",
                "fdb:",
                "--out-eps-std",
                "fdb:",
            ],
            False,
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
            True,
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
                    261022,
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
                "--date",
                "20240507",
                "--spread-compute",
                "fdb:spread_z500",
                "--ensemble",
                "fdb:ens_z500",
                "--deterministic",
                "fdb:determ_z500",
                "--clim-dir",
                "{DATA_DIR}/clustclim",
                "-N",
                "{test_dir}/NEOF",
                "--centroids",
                "fdb:",
                "--representative",
                "fdb:",
                "--output-root",
                "{test_dir}",
                "--cen-anomalies",
                "file:{test_dir}/clm_anom.grib",
                "--rep-anomalies",
                "file:{test_dir}/clr_anom.grib",
            ],
            False,
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
def test_products(
    tmpdir, monkeypatch, fdb, product, main, custom_args, pass_args, req, length
):
    monkeypatch.chdir(tmpdir)  # To avoid polluting cwd with grib templates
    shutil.copyfile(f"{TEST_DIR}/templates/{product}.yaml", f"{tmpdir}/{product}.yaml")
    with open(f"{tmpdir}/{product}.yaml", "r") as file:
        config = yaml.safe_load(file)
    config["root_dir"] = str(tmpdir)
    yaml.dump(config, open(f"{tmpdir}/{product}.yaml", "w"))
    args = [product, "-c", f"{tmpdir}/{product}.yaml"] + [
        x.format_map(
            {
                "test_dir": str(tmpdir),
                "TEST_DIR": str(TEST_DIR),
                "DATA_DIR": str(DATA_DIR),
            }
        )
        for x in custom_args
    ]
    if pass_args:
        main(args[1:])
    else:
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
