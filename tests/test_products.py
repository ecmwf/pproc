import os
import pytest
import shutil
import yaml

import pyfdb

from pproc.common.io import fdb_read_with_template
from pproc.probabilities import main as prob_main
from pproc.anomaly_probs import main as anomaly_prob_main
from pproc.ensms import main as ensms_main
from pproc.extreme import main as extreme_main
from pproc.quantiles import main as quantiles_main
from pproc.wind import main as wind_main
from pproc.clustereps.__main__ import main as clustereps_main
from helpers import NEXUS, DATA_DIR, download_test_data, populate_fdb

TEST_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(autouse=True, scope="module")
def setup_fdb(fdb):
    download_test_data(
        [
            "era_clcen_eof_mjjas.gts",
            "era_clind_mjjas.gts",
            "mjjas_eof.grd",
            "mjjas_means.grd",
            "mjjas_pcs.gts",
            "mjjas_sdv.gts",
        ],
        f"{NEXUS}/clustclim",
        f"{DATA_DIR}/clustclim",
    )
    populate_fdb(
        fdb,
        [
            "2t_ens.grib",
            "2t_clim.grib",
            "wind.grib",
            "t850.grib",
            "cluster.grib",
        ],
    )


@pytest.mark.parametrize(
    "product, main, custom_args, pass_args, req, length",
    [
        [
            "prob",
            prob_main,
            ["-d", "2024050712", "--out_prob", "fdb:"],
            False,
            {"type": "ep", "param": 131073, "step": ["12", "12-36"]},
            2,
        ],
        [
            "t850",
            anomaly_prob_main,
            ["-d", "2024050712", "--out_prob", "fdb:"],
            False,
            {
                "levtype": "pl",
                "levelist": 850,
                "type": "ep",
                "param": 131022,
                "step": [0, 12],
            },
            2,
        ],
        [
            "ensms",
            ensms_main,
            [
                "--out_eps_mean",
                "fdb:",
                "--out_eps_std",
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
                "--out_efi",
                "fdb:",
                "--out_sot",
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
                "--out_eps_mean",
                "fdb:",
                "--out_eps_std",
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
    ids=["prob", "t850", "ensms", "extreme", "quantiles", "wind", "clustereps"],
)
def test_products(
    tmpdir, monkeypatch, product, main, custom_args, pass_args, req, length
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
    _, messages = fdb_read_with_template(test_fdb, request)
    assert len(messages) == length
