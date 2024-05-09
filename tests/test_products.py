import os
import shutil
import sys
import tempfile

import requests
import pyfdb
import eccodes
import pytest
import yaml
from unittest.mock import patch

from pproc.probabilities import main as prob_main
from pproc.ensms import main as ensms_main
from pproc.extreme import main as extreme_main
from pproc.quantiles import main as quantiles_main
from pproc.wind import main as wind_main
from pproc.clustereps.__main__ import main as clustereps_main

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
NEXUS = "https://get.ecmwf.int/test-data/pproc/test-data"


def download_test_data():
    local_dir = f"{TEST_DIR}/data"
    test_files = {
        None: ["2t_ens.grib", "2t_clim.grib", "wind.grib", "cluster.grib"],
        "clustclim": [
            "era_clcen_eof_mjjas.gts",
            "era_clind_mjjas.gts",
            "mjjas_eof.grd",
            "mjjas_means.grd",
            "mjjas_pcs.gts",
            "mjjas_sdv.gts",
        ],
    }
    for dir, files in test_files.items():
        nexus_dir = NEXUS if dir is None else f"{NEXUS}/{dir}"
        dir = local_dir if dir is None else os.path.join(local_dir, dir)
        if not os.path.exists(dir):
            os.makedirs(dir)
        for filename in files:
            nexus_url = f"{nexus_dir}/{filename}"
            local_file_path = os.path.join(dir, filename)

            if not os.path.exists(local_file_path):
                session = requests.Session()
                response = session.get(nexus_url)
                if response.status_code != 200:
                    raise Exception(
                        f"Error {response.status_code} downloading data file {filename}"
                    )
                with open(local_file_path, "wb") as f:
                    f.write(response.content)


class TestProducts:
    @classmethod
    def setup_class(cls):
        download_test_data()

        cls.tmpdir = tempfile.mkdtemp()
        os.makedirs(f"{cls.tmpdir}/etc/fdb")
        os.mkdir(f"{cls.tmpdir}/fdb")
        shutil.copyfile(
            f"{TEST_DIR}/templates/fdb/schema", f"{cls.tmpdir}/etc/fdb/schema"
        )
        with open(f"{cls.tmpdir}/etc/fdb/config.yaml", "w") as f:
            f.write(
                f"""
    ---
    type: local
    engine: toc
    schema: "{cls.tmpdir}/etc/fdb/schema"
    spaces:
    - roots:
        - path: {cls.tmpdir}/fdb
    """
            )
        os.environ["FDB_HOME"] = str(cls.tmpdir)
        fdb = pyfdb.FDB()
        for file in os.listdir(f"{TEST_DIR}/data"):
            filepath = f"{TEST_DIR}/data/{file}"
            if os.path.isfile(filepath):
                reader = eccodes.FileReader(filepath)
                for msg in reader:
                    fdb.archive(msg.get_buffer())
        fdb.flush()

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmpdir)

    @pytest.mark.parametrize(
        "product, main, custom_args, pass_args",
        [
            [
                "prob",
                prob_main,
                ["-d", "2024050712", "--out_prob", "file:{test_dir}/2t_prob.grib"],
                False,
            ],
            [
                "ensms",
                ensms_main,
                [
                    "--out_eps_mean",
                    "file:{test_dir}/2t_ensms.grib",
                    "--out_eps_std",
                    "file:{test_dir}/2t_ensms.grib",
                ],
                False,
            ],
            [
                "extreme",
                extreme_main,
                [
                    "--out_efi",
                    "file:{test_dir}/2t_efi.grib",
                    "--out_sot",
                    "file:{test_dir}/2t_efi.grib",
                ],
                False,
            ],
            [
                "quantiles",
                quantiles_main,
                [
                    "--in-ens",
                    "fdb:ens",
                    "--out-quantiles",
                    "file:{test_dir}/2t_quantiles.grib",
                ],
                True,
            ],
            [
                "wind",
                wind_main,
                [
                    "--out_eps_mean",
                    "file:{test_dir}/wind.grib",
                    "--out_eps_std",
                    "file:{test_dir}/wind.grib",
                ],
                False,
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
                    "{TEST_DIR}/data/clustclim",
                    "-N",
                    "{test_dir}/NEOF",
                    "--centroids",
                    "file:{test_dir}/centroids.grib",
                    "--representative",
                    "file:{test_dir}/representative.grib",
                    "--output-root",
                    "{test_dir}",
                    "--cen-anomalies",
                    "file:{test_dir}/clm_anom.grib",
                    "--rep-anomalies",
                    "file:{test_dir}/clr_anom.grib",
                ],
                False,
            ],
        ],
        ids=["prob", "ensms", "extreme", "quantiles", "wind", "clustereps"],
    )
    def test_products(self, tmpdir, monkeypatch, product, main, custom_args, pass_args):
        test_dir = f"{tmpdir}/{product}"
        os.mkdir(test_dir)
        monkeypatch.chdir(test_dir)  # To avoid polluting cwd with grib templates
        shutil.copyfile(
            f"{TEST_DIR}/templates/{product}.yaml", f"{test_dir}/{product}.yaml"
        )
        with open(f"{test_dir}/{product}.yaml", "r") as file:
            config = yaml.safe_load(file)
        config["root_dir"] = test_dir
        yaml.dump(config, open(f"{test_dir}/{product}.yaml", "w"))
        args = [product, "-c", f"{test_dir}/{product}.yaml",] + [
            x.format_map(
                {
                    "test_dir": str(test_dir),
                    "TEST_DIR": str(TEST_DIR),
                }
            )
            for x in custom_args
        ]
        with patch("sys.argv", args):
            if pass_args:
                main(args[1:])
            else:
                main()
