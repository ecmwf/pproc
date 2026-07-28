# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import requests
from typing import List, Generator
import pytest
import shutil
import tempfile

import eccodes
import pyfdb

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
NEXUS = "https://sites.ecmwf.int/repository/pproc/test-data"
SCHEMA = os.path.join(TEST_DIR, "schema.yaml")


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
def fdb() -> Generator[pyfdb.FDB]:
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
    populate_fdb(
        temp_fdb,
        [
            "test_2t_12.grib",
        ],
        os.path.join(NEXUS, "pproc-runtime"),
    )
    populate_fdb(
        temp_fdb,
        [
            "wind.grib",
        ],
        os.path.join(NEXUS, "test-data"),
    )

    yield temp_fdb
    shutil.rmtree(tmpdir)
