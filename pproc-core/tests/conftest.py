# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from typing import Optional

import yaml

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
SCHEMA = os.path.join(TEST_DIR, "schema.yaml")


def schema(section: Optional[str] = None) -> dict:
    with open(SCHEMA, "r") as f:
        schema = yaml.safe_load(f)
    return schema if section is None else schema[section]
