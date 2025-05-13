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
import shutil
import tempfile
import requests
from typing import List, Optional
import yaml

import eccodes
import pyfdb

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
NEXUS = "https://get.ecmwf.int/test-data/pproc/test-data"
SCHEMA = os.path.join(TEST_DIR, "schema", "schema.yaml")


def download_test_data(
    test_files: List[str], dir_url: str = NEXUS, local_dir: str = DATA_DIR
) -> List[str]:

    local_files = []
    for file in test_files:
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        nexus_url = f"{dir_url}/{file}"
        local_file_path = os.path.join(local_dir, file)

        if not os.path.exists(local_file_path):
            session = requests.Session()
            response = session.get(nexus_url)
            if response.status_code != 200:
                raise Exception(
                    f"Error {response.status_code} downloading data file {file}"
                )
            with open(local_file_path, "wb") as f:
                f.write(response.content)
        local_files.append(local_file_path)
    return local_files


def populate_fdb(
    fdb,
    test_files: List[str],
    dir_url: str = NEXUS,
    local_dir: str = DATA_DIR,
):
    data_files = download_test_data(test_files, dir_url, local_dir)
    for filepath in data_files:
        if os.path.isfile(filepath):
            reader = eccodes.FileReader(filepath)
            for msg in reader:
                fdb.archive(msg.get_buffer())
    fdb.flush()


@pytest.fixture(scope="session")
def fdb() -> pyfdb.FDB:
    tmpdir = tempfile.mkdtemp()
    print("Using temporary directory", tmpdir)
    os.makedirs(f"{tmpdir}/etc/fdb")
    os.mkdir(f"{tmpdir}/fdb")
    shutil.copyfile(f"{TEST_DIR}/templates/fdb/schema", f"{tmpdir}/etc/fdb/schema")
    with open(f"{tmpdir}/etc/fdb/config.yaml", "w") as f:
        f.write(
            f"""
---
type: local
engine: toc
schema: "{tmpdir}/etc/fdb/schema"
spaces:
- roots:
    - path: {tmpdir}/fdb
"""
        )
    os.environ["FDB_HOME"] = str(tmpdir)
    os.environ["FDB_HANDLE_LUSTRE_STRIPE"] = "0"
    temp_fdb = pyfdb.FDB()
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
        temp_fdb,
        [
            "2t_ens.grib",
            "2t_clim.grib",
            "wind.grib",
            "t850.grib",
            "cluster.grib",
            "has_missing.grib",
            "thermo.grib",
        ],
    )

    yield temp_fdb
    shutil.rmtree(tmpdir)


def schema(section: Optional[str] = None) -> dict:
    with open(SCHEMA, "r") as f:
        schema = yaml.safe_load(f)
    return schema if section is None else schema[section]
