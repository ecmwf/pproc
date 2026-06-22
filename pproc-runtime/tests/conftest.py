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
from typing import List

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
NEXUS = "https://sites.ecmwf.int/repository/pproc/test-data/pproc-runtime"
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

download_test_data(
    [
        "test_2t_12.grib",
    ],
    f"{NEXUS}",
    f"{DATA_DIR}",
)